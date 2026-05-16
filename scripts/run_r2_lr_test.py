"""Quick R2 lr triple-test (DDQN, norm=off, seed=42, 100k timesteps each).

R2 (raw dollar P&L) per-bar magnitude is ~$1-30 -- about 10^4-10^5 x larger
than R1's ~1e-4/bar log-return. With reward normalization OFF (confirmed
necessary for R1 in the prior A/B), R2 Q-values will be at a wildly different
scale than R1's. Using R1's lr=0.0045 directly on R2 risks gradient explosion
/ NaN.

This test rides 3 candidate lrs:
  - 4.5e-7  (R1 lr / 10^4, scale-matched)
  - 1e-6    (slightly more permissive)
  - 1e-5    (test if mid-range still trains)

Criteria for "viable":
  1. No NaN loss / Q-value
  2. trades>0 in at least 1 val eval
  3. Final equity not absurdly negative (no obvious divergence)

Output: runs/r2_lr_test/lr_<value>/  + runs/r2_lr_test/r2_lr_summary.csv
"""
from __future__ import annotations

import copy
import csv
import json
import math
import sys
import time
from pathlib import Path

import yaml

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))


def main() -> None:
    cfg_base = yaml.safe_load((WORKSPACE_ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg_base["env"]["reward"]["mode"] = "r2"
    cfg_base["env"]["reward"]["normalize"] = False
    cfg_base["train"]["seed"] = 42
    cfg_base["train"]["total_timesteps"] = 100_000
    cfg_base["train"]["eval_every_sessions"] = 22
    cfg_base["train"]["early_stop_patience"] = 99   # no early-stop for the test
    cfg_base["run"]["output_root"] = "runs/r2_lr_test"

    lrs = [4.5e-7, 1e-6, 1e-5]

    from train import train_ddqn

    summary: list[dict] = []
    for lr in lrs:
        tag = f"lr_{lr:.0e}".replace("-0", "-")
        print(f"\n{'='*72}")
        print(f"R2 lr test: lr={lr}  run={tag}")
        print(f"{'='*72}")
        cfg = copy.deepcopy(cfg_base)
        cfg["dqn"]["lr"] = float(lr)
        cfg["run"]["run_name"] = tag

        t0 = time.time()
        try:
            test_m = train_ddqn(cfg)
            err = None
        except Exception as e:
            test_m = {}
            err = f"{type(e).__name__}: {e}"
        elapsed = time.time() - t0

        # Parse val rows
        run_dir = WORKSPACE_ROOT / "runs" / "r2_lr_test" / tag
        metrics_csv = run_dir / "metrics.csv"
        max_val_trades = 0
        n_val_evals = 0
        nan_seen = False
        if metrics_csv.exists():
            with metrics_csv.open(encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for r in reader:
                    if r.get("phase") == "val":
                        n_val_evals += 1
                        t = int(r["trades"])
                        if t > max_val_trades:
                            max_val_trades = t
                        try:
                            if math.isnan(float(r["total_return"])):
                                nan_seen = True
                        except ValueError:
                            nan_seen = True

        bi_path = run_dir / "best_info.json"
        bi = json.loads(bi_path.read_text(encoding="utf-8")) if bi_path.exists() else {}

        summary.append({
            "lr": lr,
            "tag": tag,
            "wall_seconds": round(elapsed, 1),
            "error": err or "",
            "nan_in_val": nan_seen,
            "n_val_evals": n_val_evals,
            "max_val_trades": max_val_trades,
            "val_best_total_return": bi.get("val_total_return", float("nan")),
            "val_best_trades": bi.get("val_trades", 0),
            "val_best_final_equity": bi.get("val_final_equity", float("nan")),
            "test_total_return": test_m.get("total_return", float("nan")),
            "test_trades": test_m.get("trades", 0),
            "test_final_equity": test_m.get("final_equity", float("nan")),
        })

    out_root = (WORKSPACE_ROOT / "runs" / "r2_lr_test").resolve()
    out_csv = out_root / "r2_lr_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\nSummary -> {out_csv}")
    print()
    print(f"{'lr':>10} {'wall':>6} {'nan':>5} {'val_evals':>10} {'max_tr':>8} {'best_eq':>10} {'test_tr':>8} {'test_eq':>10}")
    for r in summary:
        nan_tag = "YES" if r["nan_in_val"] else "no"
        eq_str = f"${r['val_best_final_equity']:.0f}" if not math.isnan(r['val_best_final_equity']) else "NaN"
        teq_str = f"${r['test_final_equity']:.0f}" if not math.isnan(r['test_final_equity']) else "NaN"
        print(f"{r['lr']:>10.1e} {r['wall_seconds']:>6.0f} {nan_tag:>5} "
              f"{r['n_val_evals']:>10} {r['max_val_trades']:>8} "
              f"{eq_str:>10} {r['test_trades']:>8} {teq_str:>10}")

    # Verdict: pick the lr with max_val_trades > 0 AND no NaN AND best final_equity
    viable = [r for r in summary if r["max_val_trades"] > 0 and not r["nan_in_val"]]
    if viable:
        winner = max(viable, key=lambda r: r["val_best_final_equity"]
                     if not math.isnan(r["val_best_final_equity"]) else float("-inf"))
        print(f"\nRECOMMENDED R2 lr: {winner['lr']}")
        print(f"  (val best final_equity = ${winner['val_best_final_equity']:.2f}, "
              f"trades = {winner['val_best_trades']})")
    else:
        print("\nNo viable lr found. Need wider sweep.")


if __name__ == "__main__":
    main()
