"""Quick R1 lr A/B (DDQN, norm=off, seed=42, 100k timesteps each).

Hypothesis (user-asked 2026-05-16): Run B from the norm A/B used R1 with
lr=0.0045 and got noisy training (trades swung 20<->2599 across evals) and
poor final val (best ret=+0.004 only). Maybe a smaller lr stabilises learning.

This A/B tests:
  - lr=0.0045 (current Run B baseline)
  - lr=1e-5   (matched to R2's chosen lr)

Criteria for "better":
  1. Higher val_best_total_return at 100k
  2. More stable trade count across evals (smaller swing)
  3. No NaN

Output: runs/r1_lr_test/lr_<value>/  + runs/r1_lr_test/r1_lr_summary.csv
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
    cfg_base["env"]["reward"]["mode"] = "r1"
    cfg_base["env"]["reward"]["normalize"] = False
    cfg_base["train"]["seed"] = 42
    cfg_base["train"]["total_timesteps"] = 100_000
    cfg_base["train"]["eval_every_sessions"] = 22
    cfg_base["train"]["early_stop_patience"] = 99   # no early-stop in the test
    cfg_base["run"]["output_root"] = "runs/r1_lr_test"

    lrs = [0.0045, 1e-5]

    from train import train_ddqn

    summary: list[dict] = []
    for lr in lrs:
        tag = f"lr_{lr:.0e}".replace("-0", "-").replace("+0", "+")
        # special-case the baseline for readability
        if lr == 0.0045:
            tag = "lr_0.0045"
        print(f"\n{'='*72}")
        print(f"R1 lr test: lr={lr}  run={tag}")
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

        run_dir = WORKSPACE_ROOT / "runs" / "r1_lr_test" / tag
        metrics_csv = run_dir / "metrics.csv"
        val_trade_list: list[int] = []
        nan_seen = False
        if metrics_csv.exists():
            with metrics_csv.open(encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for r in reader:
                    if r.get("phase") == "val":
                        try:
                            val_trade_list.append(int(r["trades"]))
                            if math.isnan(float(r["total_return"])):
                                nan_seen = True
                        except ValueError:
                            nan_seen = True

        bi_path = run_dir / "best_info.json"
        bi = json.loads(bi_path.read_text(encoding="utf-8")) if bi_path.exists() else {}

        trade_swing = (max(val_trade_list) - min(val_trade_list)) if val_trade_list else 0

        summary.append({
            "lr": lr,
            "tag": tag,
            "wall_seconds": round(elapsed, 1),
            "error": err or "",
            "nan_in_val": nan_seen,
            "n_val_evals": len(val_trade_list),
            "min_val_trades": min(val_trade_list) if val_trade_list else 0,
            "max_val_trades": max(val_trade_list) if val_trade_list else 0,
            "trade_swing": trade_swing,
            "val_best_total_return": bi.get("val_total_return", float("nan")),
            "val_best_trades": bi.get("val_trades", 0),
            "val_best_final_equity": bi.get("val_final_equity", float("nan")),
            "test_total_return": test_m.get("total_return", float("nan")),
            "test_trades": test_m.get("trades", 0),
            "test_final_equity": test_m.get("final_equity", float("nan")),
        })

    out_root = (WORKSPACE_ROOT / "runs" / "r1_lr_test").resolve()
    out_csv = out_root / "r1_lr_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\nSummary -> {out_csv}")
    print()
    print(f"{'lr':>10} {'wall':>6} {'evals':>6} {'tr_min':>7} {'tr_max':>7} {'swing':>7} {'val_best_ret':>13} {'val_best_eq':>12} {'test_ret':>9} {'test_eq':>10}")
    for r in summary:
        eq_str = f"${r['val_best_final_equity']:.2f}" if not math.isnan(r['val_best_final_equity']) else "NaN"
        teq_str = f"${r['test_final_equity']:.2f}" if not math.isnan(r['test_final_equity']) else "NaN"
        print(f"{r['lr']:>10.0e} {r['wall_seconds']:>6.0f} {r['n_val_evals']:>6} "
              f"{r['min_val_trades']:>7} {r['max_val_trades']:>7} {r['trade_swing']:>7} "
              f"{r['val_best_total_return']:>+13.4f} {eq_str:>12} "
              f"{r['test_total_return']:>+9.4f} {teq_str:>10}")

    # Verdict: pick higher val_best_total_return
    viable = [r for r in summary if not r["nan_in_val"] and r["max_val_trades"] > 0]
    if viable:
        winner = max(viable, key=lambda r: r["val_best_total_return"]
                     if not math.isnan(r["val_best_total_return"]) else float("-inf"))
        print(f"\nRECOMMENDED R1 lr: {winner['lr']}")
        print(f"  val best total_return = {winner['val_best_total_return']:+.6f}")
        print(f"  val best final_equity = ${winner['val_best_final_equity']:.2f}")
        print(f"  trade swing across evals = {winner['trade_swing']} (lower=more stable)")
    else:
        print("\nNo viable lr found.")


if __name__ == "__main__":
    main()
