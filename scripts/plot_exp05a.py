"""Plot Exp 0.5a (γ-sweep) results.

Generates:
  - gamma_summary.png      — 2x2 grid: collapse rate, mean test return, mean Sortino, mean trades
  - test_metrics_by_run.png — per-run scatter of test_return and Sortino, colored by γ
  - equity_curves.png      — per-run equity over test sessions, colored by γ
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = WORKSPACE_ROOT / "runs" / "exp05a"

GAMMA_COLORS = {0.3: "#1f77b4", 0.5: "#2ca02c", 0.9: "#ff7f0e", 0.99: "#d62728"}


def load_rows() -> list[dict]:
    p = EXP_DIR / "exp05a_summary.csv"
    return list(csv.DictReader(p.open(encoding="utf-8")))


def plot_gamma_summary(rows: list[dict]) -> None:
    groups = defaultdict(list)
    for r in rows:
        groups[float(r["gamma"])].append(r)

    gammas = sorted(groups.keys())
    n_collapse = [sum(1 for r in groups[g] if int(float(r["val_trades"])) < 50) for g in gammas]
    test_rets = [[float(r["test_total_return"]) for r in groups[g]] for g in gammas]
    sortinos = [[float(r["test_sortino"]) for r in groups[g]] for g in gammas]
    trades = [[float(r["test_trades"]) for r in groups[g]] for g in gammas]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. Collapse rate
    ax = axes[0, 0]
    colors = [GAMMA_COLORS[g] for g in gammas]
    ax.bar([str(g) for g in gammas], [c / len(groups[g]) for c, g in zip(n_collapse, gammas)], color=colors)
    ax.set_title("Collapse rate (val_trades < 50)")
    ax.set_ylabel("Fraction of seeds collapsed")
    ax.set_xlabel("γ")
    ax.set_ylim(0, 1)
    for i, (g, c) in enumerate(zip(gammas, n_collapse)):
        ax.text(i, c / len(groups[g]) + 0.02, f"{c}/{len(groups[g])}", ha="center", fontsize=10)

    # 2. Test return (mean ± std)
    ax = axes[0, 1]
    means = [np.mean(t) for t in test_rets]
    stds = [np.std(t) for t in test_rets]
    ax.bar([str(g) for g in gammas], means, yerr=stds, color=colors, capsize=6)
    ax.set_title("Test total_return (mean ± std across seeds)")
    ax.set_ylabel("Total return")
    ax.set_xlabel("γ")
    ax.axhline(0, color="k", linewidth=0.5)

    # 3. Sortino
    ax = axes[1, 0]
    s_means = [np.mean(s) for s in sortinos]
    s_stds = [np.std(s) for s in sortinos]
    ax.bar([str(g) for g in gammas], s_means, yerr=s_stds, color=colors, capsize=6)
    ax.set_title("Test Sortino (mean ± std)")
    ax.set_ylabel("Sortino")
    ax.set_xlabel("γ")
    ax.axhline(0, color="k", linewidth=0.5)

    # 4. Trades
    ax = axes[1, 1]
    t_means = [np.mean(t) for t in trades]
    ax.bar([str(g) for g in gammas], t_means, color=colors)
    ax.set_title("Test trades (mean across seeds)")
    ax.set_ylabel("# trades")
    ax.set_xlabel("γ")
    ax.axhline(50, color="r", linestyle="--", linewidth=1, label="Collapse threshold (50)")
    ax.legend()

    plt.suptitle("Exp 0.5a — DDQN γ-sweep on R4 (3 seeds per γ, 300k steps)", fontsize=13)
    plt.tight_layout()
    out = EXP_DIR / "gamma_summary.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")


def plot_per_run(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: test_return scatter
    ax = axes[0]
    for r in rows:
        g = float(r["gamma"])
        seed = int(r["seed"])
        ax.scatter(g, float(r["test_total_return"]), color=GAMMA_COLORS[g], s=80, alpha=0.8,
                   edgecolors="k", linewidth=0.5)
        ax.annotate(f"s{seed}", (g, float(r["test_total_return"])),
                    xytext=(8, 0), textcoords="offset points", fontsize=8, alpha=0.7)
    ax.set_xlabel("γ")
    ax.set_ylabel("Test total_return")
    ax.set_title("Test return per (γ, seed)")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xticks(sorted(GAMMA_COLORS.keys()))
    ax.grid(True, alpha=0.3)

    # Right: Sortino scatter
    ax = axes[1]
    for r in rows:
        g = float(r["gamma"])
        seed = int(r["seed"])
        ax.scatter(g, float(r["test_sortino"]), color=GAMMA_COLORS[g], s=80, alpha=0.8,
                   edgecolors="k", linewidth=0.5)
        ax.annotate(f"s{seed}", (g, float(r["test_sortino"])),
                    xytext=(8, 0), textcoords="offset points", fontsize=8, alpha=0.7)
    ax.set_xlabel("γ")
    ax.set_ylabel("Test Sortino")
    ax.set_title("Test Sortino per (γ, seed)")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xticks(sorted(GAMMA_COLORS.keys()))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = EXP_DIR / "test_metrics_by_run.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")


def plot_equity_curves(rows: list[dict]) -> None:
    """Plot equity curves from trades.csv for each run (test phase)."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
    axes = axes.flatten()
    gammas = sorted({float(r["gamma"]) for r in rows})

    for ax, gamma in zip(axes, gammas):
        rows_g = [r for r in rows if abs(float(r["gamma"]) - gamma) < 1e-9]
        for r in sorted(rows_g, key=lambda x: int(x["seed"])):
            run_name = r["run_name"]
            trades_path = EXP_DIR / run_name / "trades.csv"
            if not trades_path.exists():
                continue
            try:
                df = pd.read_csv(trades_path)
                test_df = df[df["phase"] == "test"]
                if len(test_df) == 0:
                    continue
                # cumulative log P&L from each trade
                cum_pnl = test_df["pnl_log"].cumsum().values
                equity = 10000 * np.exp(cum_pnl)
                ax.plot(equity, alpha=0.85, linewidth=1.2, label=f"s{r['seed']} (ret={float(r['test_total_return']):+.2%}, n={len(test_df)})")
            except Exception as e:
                print(f"  warn: could not plot {run_name}: {e}")
        ax.axhline(10000, color="k", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_title(f"γ = {gamma}")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Equity (USD)")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Exp 0.5a — Test-phase equity curves (one line = one seed)", fontsize=13)
    plt.tight_layout()
    out = EXP_DIR / "equity_curves.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")


def main():
    rows = load_rows()
    print(f"Loaded {len(rows)} rows from {EXP_DIR / 'exp05a_summary.csv'}")
    plot_gamma_summary(rows)
    plot_per_run(rows)
    plot_equity_curves(rows)
    print("Done.")


if __name__ == "__main__":
    main()
