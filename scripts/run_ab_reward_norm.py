"""A/B test: reward_normalize ON vs OFF for DDQN R1 seed=42.

Hypothesis (user-suggested 2026-05-16): the running-std reward normalizer
amplifies R1's tiny per-bar log-return (~1e-4) by ~10^4 early in training when
the normalizer std is still small, biasing the Q-network toward the "flat"
attractor (Q(flat)=0 dominates all noisy trading Q-values). Disabling
normalization keeps R1 at its natural scale and *may* let the agent learn
beyond the do-nothing policy.

This is a focused 2-run experiment, not a full sweep. Run results go to:
  runs/ab_reward_norm/A_norm_on/
  runs/ab_reward_norm/B_norm_off/
plus runs/ab_reward_norm/ab_summary.csv with side-by-side val curves.

The two runs are sequential (not parallel) to avoid GPU contention.
"""
from __future__ import annotations

import copy
import csv
import json
import sys
import time
from pathlib import Path

import yaml

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))


def main() -> None:
    cfg_base = yaml.safe_load((WORKSPACE_ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg_base["env"]["reward"]["mode"] = "r1"
    cfg_base["train"]["seed"] = 42
    cfg_base["run"]["output_root"] = "runs/ab_reward_norm"

    out_root = (WORKSPACE_ROOT / "runs" / "ab_reward_norm").resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    runs = [
        ("A_norm_on", True),
        ("B_norm_off", False),
    ]

    from train import train_ddqn

    summary_rows: list[dict] = []
    val_curves: dict[str, list[dict]] = {}
    for run_name, normalize in runs:
        print(f"\n{'='*72}")
        print(f"RUN {run_name}  normalize={normalize}  reward_mode=r1  seed=42")
        print(f"{'='*72}")
        cfg = copy.deepcopy(cfg_base)
        cfg["env"]["reward"]["normalize"] = normalize
        cfg["run"]["run_name"] = run_name
        t0 = time.time()
        test_m = train_ddqn(cfg)
        elapsed = time.time() - t0
        run_dir = out_root / run_name
        bi_path = run_dir / "best_info.json"
        bi = json.loads(bi_path.read_text(encoding="utf-8")) if bi_path.exists() else {}

        # Parse val rows from metrics.csv so we can see if either run escaped flat.
        metrics_csv = run_dir / "metrics.csv"
        val_rows: list[dict] = []
        if metrics_csv.exists():
            with metrics_csv.open(encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for r in reader:
                    if r.get("phase") == "val":
                        val_rows.append({
                            "episode": int(r["episode"]),
                            "step": int(r["global_step"]),
                            "total_return": float(r["total_return"]),
                            "trades": int(r["trades"]),
                            "final_equity": float(r["final_equity"]),
                            "max_dd_dollar": float(r["max_dd_dollar"]),
                            "ruin_rate": float(r["ruin_rate"]),
                        })
        val_curves[run_name] = val_rows
        any_trades_in_val = max((r["trades"] for r in val_rows), default=0)

        row = {
            "run": run_name,
            "normalize": normalize,
            "wall_seconds": round(elapsed, 1),
            "n_val_evals": len(val_rows),
            "val_evals_with_trades": sum(1 for r in val_rows if r["trades"] > 0),
            "val_max_trades_seen": any_trades_in_val,
            "val_best_total_return": bi.get("val_total_return", float("nan")),
            "val_best_trades": bi.get("val_trades", 0),
            "val_best_final_equity": bi.get("val_final_equity", float("nan")),
            "best_episode": bi.get("episode", -1),
            "test_total_return": test_m.get("total_return", float("nan")),
            "test_trades": test_m.get("trades", 0),
            "test_final_equity": test_m.get("final_equity", float("nan")),
            "test_sharpe": test_m.get("sharpe", float("nan")),
            "test_sortino": test_m.get("sortino", float("nan")),
            "test_mdd_dollar": test_m.get("max_dd_dollar", float("nan")),
        }
        summary_rows.append(row)

    # Dump val curves for plotting later
    (out_root / "val_curves.json").write_text(
        json.dumps(val_curves, indent=2), encoding="utf-8"
    )

    # Side-by-side summary
    summary_path = out_root / "ab_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nA/B summary -> {summary_path}")
    print()
    print(f"{'run':<14} {'norm':>6} {'evals':>6} {'evals_w/_tr':>12} {'val_best_eq':>12} {'test_trades':>11} {'test_eq':>10} {'test_ret':>9}")
    for r in summary_rows:
        print(f"{r['run']:<14} {str(r['normalize']):>6} {r['n_val_evals']:>6} "
              f"{r['val_evals_with_trades']:>12} ${r['val_best_final_equity']:>10.2f} "
              f"{r['test_trades']:>11} ${r['test_final_equity']:>8.2f} "
              f"{r['test_total_return']:>+9.4f}")

    # Verdict
    print()
    a = next(r for r in summary_rows if r["normalize"] is True)
    b = next(r for r in summary_rows if r["normalize"] is False)
    print(f"Verdict:")
    print(f"  norm=ON  reached val trades>0 in {a['val_evals_with_trades']}/{a['n_val_evals']} evals (best test trades={a['test_trades']})")
    print(f"  norm=OFF reached val trades>0 in {b['val_evals_with_trades']}/{b['n_val_evals']} evals (best test trades={b['test_trades']})")
    if a["val_evals_with_trades"] == 0 and b["val_evals_with_trades"] == 0:
        print(f"  -> Both collapsed to flat. Reward normalization is NOT the bottleneck for R1.")
    elif b["val_evals_with_trades"] > a["val_evals_with_trades"]:
        print(f"  -> norm=OFF escapes flat more often. Confirms hypothesis.")
    elif a["val_evals_with_trades"] > b["val_evals_with_trades"]:
        print(f"  -> norm=ON escapes flat more often. Hypothesis NOT confirmed; current default is fine.")
    else:
        print(f"  -> Both similar. Reward norm not decisive on R1.")


if __name__ == "__main__":
    main()
