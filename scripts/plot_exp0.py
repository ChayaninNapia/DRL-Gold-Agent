"""Generate Exp 0 v2 plots for PROPOSAL.md embedding.

4 plots saved to runs/exp0/plots/:
  1. reward_x_algo_heatmap.png — test_total_return heatmap (3 rewards × 3 algos, mean over seeds)
  2. test_return_by_run.png    — bar chart of all 27 runs, color-coded by algo, hatched by reward
  3. equity_curves_top.png     — top 4 runs equity curves on test set (reconstructed from trades.csv)
  4. collapse_audit.png        — bar chart showing test_trades per (reward, algo, seed); flat runs = trades=0

Uses matplotlib only. Reads runs/exp0/exp0_summary.csv + per-run trades.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display; just save PNGs
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
EXP0_DIR = WORKSPACE_ROOT / "runs" / "exp0"
PLOTS_DIR = EXP0_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

REWARDS = ["r1", "r2", "r4"]
ALGOS = ["ddqn", "a2c", "ppo"]
SEEDS = [42, 1337, 2026]
ALGO_COLORS = {"ddqn": "#1f77b4", "a2c": "#ff7f0e", "ppo": "#2ca02c"}
REWARD_LABELS = {"r1": "R1 (log-return)", "r2": "R2 ($ P&L)", "r4": "R4 ($ - DD penalty)"}


def load_summary() -> pd.DataFrame:
    df = pd.read_csv(EXP0_DIR / "exp0_summary.csv")
    return df


# ============================================================
# Plot 1 — reward x algo heatmap (mean test_total_return)
# ============================================================
def plot_heatmap(df: pd.DataFrame) -> None:
    mat = np.zeros((3, 3))
    for i, r in enumerate(REWARDS):
        for j, a in enumerate(ALGOS):
            sub = df[(df["reward"] == r) & (df["algo"] == a)]
            mat[i, j] = sub["test_total_return"].mean() * 100  # to percent

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    # Diverging colormap centered at 0
    vmax = float(np.max(np.abs(mat)))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([a.upper() for a in ALGOS], fontsize=11)
    ax.set_yticklabels([REWARD_LABELS[r] for r in REWARDS], fontsize=11)
    ax.set_xlabel("Algorithm", fontsize=12)
    ax.set_ylabel("Reward variant", fontsize=12)
    ax.set_title("Exp 0 v2 — Mean test total_return (%) across 3 seeds\n"
                 "Green = profitable; red = lost capital; pale = do-nothing flat",
                 fontsize=12)
    for i in range(3):
        for j in range(3):
            val = mat[i, j]
            color = "white" if abs(val) > vmax * 0.5 else "black"
            ax.text(j, i, f"{val:+.2f}%", ha="center", va="center",
                    color=color, fontsize=12, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, label="Mean test total_return (%)")
    plt.tight_layout()
    out = PLOTS_DIR / "reward_x_algo_heatmap.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")


# ============================================================
# Plot 2 — per-run test return bar chart, grouped by reward, colored by algo
# ============================================================
def plot_per_run_bar(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.5))
    bar_w = 0.25
    x = np.arange(len(REWARDS))  # 3 reward groups

    for ai, algo in enumerate(ALGOS):
        for si, seed in enumerate(SEEDS):
            heights = []
            for r in REWARDS:
                row = df[(df["reward"] == r) & (df["algo"] == algo) & (df["seed"] == seed)]
                heights.append(row["test_total_return"].iloc[0] * 100)
            offset = (ai - 1) * bar_w
            xs = x + offset
            # Each algo group has 3 bars (one per seed) stacked horizontally with light shading
            # We'll cheat: plot 3 bars per algo group, indexed by seed
            x_seed = xs + (si - 1) * (bar_w / 3.2)
            ax.bar(x_seed, heights, width=bar_w / 3.5, color=ALGO_COLORS[algo],
                   alpha=0.55 + 0.225 * si,  # 0.55, 0.775, 1.0
                   edgecolor="black", linewidth=0.4,
                   label=f"{algo.upper()} s{seed}" if ai == 0 and si == 0 else None)

    # Legend: 3 algos × 3 seeds = 9 bars per reward — too many for normal legend; show algo legend only
    handles = [plt.Rectangle((0, 0), 1, 1, color=ALGO_COLORS[a], alpha=0.78,
                             edgecolor="black", linewidth=0.4) for a in ALGOS]
    ax.legend(handles, [a.upper() for a in ALGOS], title="Algorithm", loc="lower left", fontsize=10)

    ax.axhline(0.0, color="black", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([REWARD_LABELS[r] for r in REWARDS], fontsize=11)
    ax.set_ylabel("Test total_return (%)", fontsize=12)
    ax.set_title("Exp 0 v2 — Test return per run (3 seeds × 3 algos × 3 rewards = 27 runs)\n"
                 "Bar alpha encodes seed (light=42, mid=1337, dark=2026)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "test_return_by_run.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")


# ============================================================
# Plot 3 — equity curves of top runs on test set
# ============================================================
def plot_equity_curves(df: pd.DataFrame) -> None:
    # Pick top 4 by test_total_return + the long baseline if we have it as a sanity reference.
    top = df.sort_values("test_total_return", ascending=False).head(4)

    fig, ax = plt.subplots(figsize=(13, 6))
    capital0 = 10_000.0

    for _, row in top.iterrows():
        rn = row["run_name"]
        trades_csv = EXP0_DIR / rn / "trades.csv"
        if not trades_csv.exists():
            continue
        td = pd.read_csv(trades_csv)
        td = td[td["phase"] == "test"]
        if td.empty:
            continue
        # Cumulative equity using pnl_log per closed trade, multiplied through capital0.
        # equity_after_trade = capital0 * exp(cumsum(pnl_log))
        eq = capital0 * np.exp(td["pnl_log"].cumsum())
        eq = pd.concat([pd.Series([capital0]), eq], ignore_index=True)
        ax.plot(
            range(len(eq)), eq,
            label=f"{rn} (ret={row['test_total_return']*100:+.2f}%, Sh={row['test_sharpe']:+.2f})",
            linewidth=1.6,
        )

    ax.axhline(capital0, color="black", linestyle="--", linewidth=0.7, alpha=0.5,
               label=f"Capital0 = ${capital0:,.0f}")
    ax.set_xlabel("Trade number (within test set, 75 days)", fontsize=12)
    ax.set_ylabel("Equity ($)", fontsize=12)
    ax.set_title("Exp 0 v2 — Equity curves of top 4 runs (test set, trade-indexed)",
                 fontsize=12)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "equity_curves_top.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")


# ============================================================
# Plot 4 — collapse audit: test_trades per (reward, algo, seed)
# ============================================================
def plot_collapse_audit(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.5))
    bar_w = 0.10
    x = np.arange(len(REWARDS))

    for ai, algo in enumerate(ALGOS):
        for si, seed in enumerate(SEEDS):
            heights = []
            for r in REWARDS:
                row = df[(df["reward"] == r) & (df["algo"] == algo) & (df["seed"] == seed)]
                heights.append(row["test_trades"].iloc[0])
            x_pos = x + (ai - 1) * 0.30 + (si - 1) * (bar_w * 1.05)
            ax.bar(x_pos, heights, width=bar_w, color=ALGO_COLORS[algo],
                   alpha=0.55 + 0.225 * si,
                   edgecolor="black", linewidth=0.4)
            # Annotate flat runs with red "x"
            for xi, h in enumerate(heights):
                if h == 0:
                    ax.text(x_pos[xi], 50, "FLAT", ha="center", va="bottom",
                            fontsize=7, color="red", fontweight="bold", rotation=90)

    handles = [plt.Rectangle((0, 0), 1, 1, color=ALGO_COLORS[a], alpha=0.78,
                             edgecolor="black", linewidth=0.4) for a in ALGOS]
    ax.legend(handles, [a.upper() for a in ALGOS], title="Algorithm", loc="upper right", fontsize=10)

    ax.set_yscale("symlog", linthresh=10)
    ax.set_xticks(x)
    ax.set_xticklabels([REWARD_LABELS[r] for r in REWARDS], fontsize=11)
    ax.set_ylabel("Test trades count (symlog)", fontsize=12)
    ax.set_title("Exp 0 v2 — Trades per run on test set; FLAT label = collapse to do-nothing\n"
                 "Bar alpha encodes seed (light=42, mid=1337, dark=2026)", fontsize=12)
    ax.grid(axis="y", alpha=0.3, which="both")
    plt.tight_layout()
    out = PLOTS_DIR / "collapse_audit.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")


# ============================================================
# Plot 5 — val_total_return ranking with mean rank annotation
# ============================================================
def plot_ranking(df: pd.DataFrame) -> None:
    mean_table = df.groupby(["reward", "algo"])["val_total_return"].mean().unstack()
    mean_table = mean_table.reindex(index=REWARDS, columns=ALGOS) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_w = 0.25
    x = np.arange(len(REWARDS))
    for ai, algo in enumerate(ALGOS):
        vals = mean_table[algo].values
        offset = (ai - 1) * bar_w
        bars = ax.bar(x + offset, vals, width=bar_w, color=ALGO_COLORS[algo],
                      alpha=0.85, edgecolor="black", linewidth=0.6, label=algo.upper())
        for bi, b in enumerate(bars):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{vals[bi]:+.2f}%", ha="center",
                    va="bottom" if vals[bi] >= 0 else "top",
                    fontsize=8, fontweight="bold")

    ax.axhline(0.0, color="black", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([REWARD_LABELS[r] for r in REWARDS], fontsize=11)
    ax.set_ylabel("Mean val_total_return (%) across 3 seeds", fontsize=11)
    ax.set_title("Exp 0 v2 ranking — val metric drives best-checkpoint selection\n"
                 "WINNER: R4 (mean rank 1.67 across DDQN/A2C/PPO)", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "exp0_ranking.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    df = load_summary()
    plot_heatmap(df)
    plot_ranking(df)
    plot_per_run_bar(df)
    plot_equity_curves(df)
    plot_collapse_audit(df)
    print(f"\nAll plots saved to {PLOTS_DIR}")
