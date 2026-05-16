"""Quick R2 lr sweep for A2C and PPO (norm=off, seed=42, 100k timesteps each).

A2C/PPO are on-policy policy-gradient methods; reward magnitude enters the
gradient through advantage = return - V(s) directly (not absorbed by a Q-net
like DDQN). So R2's ~$10/bar scale needs even more lr reduction than DDQN.

DDQN found R2 lr=1e-5 worked (vs default 0.0045 -> ~450x reduction).
We sweep A2C/PPO at lr {1e-5, 1e-6, 1e-7} to find the band.

Output: runs/a2c_ppo_r2_lr_test/<algo>_lr_<value>/ + a2c_ppo_r2_lr_summary.csv
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
    cfg_base["train"]["early_stop_patience"] = 99
    cfg_base["run"]["output_root"] = "runs/a2c_ppo_r2_lr_test"

    sweep = [
        ("a2c", 1e-5),
        ("a2c", 1e-6),
        ("a2c", 1e-7),
        ("ppo", 1e-5),
        ("ppo", 1e-6),
        ("ppo", 1e-7),
    ]

    from train_a2c import train_a2c
    from train_ppo import train_ppo
    trainers = {"a2c": train_a2c, "ppo": train_ppo}

    summary: list[dict] = []
    for algo, lr in sweep:
        tag = f"{algo}_lr_{lr:.0e}".replace("-0", "-")
        print(f"\n{'='*72}")
        print(f"R2 lr test ({algo.upper()}): lr={lr}  run={tag}")
        print(f"{'='*72}")
        cfg = copy.deepcopy(cfg_base)
        cfg[algo]["lr"] = float(lr)
        cfg["run"]["run_name"] = tag

        t0 = time.time()
        try:
            test_m = trainers[algo](cfg)
            err = None
        except Exception as e:
            test_m = {}
            err = f"{type(e).__name__}: {e}"
        elapsed = time.time() - t0

        run_dir = WORKSPACE_ROOT / "runs" / "a2c_ppo_r2_lr_test" / tag
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
            "algo": algo,
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

    out_root = (WORKSPACE_ROOT / "runs" / "a2c_ppo_r2_lr_test").resolve()
    out_csv = out_root / "a2c_ppo_r2_lr_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\nSummary -> {out_csv}")
    print()
    print(f"{'algo':<5} {'lr':>8} {'nan':>4} {'tr_min':>7} {'tr_max':>7} "
          f"{'val_best_ret':>13} {'val_best_eq':>12} {'test_ret':>9} {'test_eq':>10}")
    for r in summary:
        eq_str = f"${r['val_best_final_equity']:.2f}" if not math.isnan(r['val_best_final_equity']) else "NaN"
        teq_str = f"${r['test_final_equity']:.2f}" if not math.isnan(r['test_final_equity']) else "NaN"
        nan_tag = "YES" if r["nan_in_val"] else "no"
        print(f"{r['algo']:<5} {r['lr']:>8.0e} {nan_tag:>4} "
              f"{r['min_val_trades']:>7} {r['max_val_trades']:>7} "
              f"{r['val_best_total_return']:>+13.4f} {eq_str:>12} "
              f"{r['test_total_return']:>+9.4f} {teq_str:>10}")

    # Per-algo winner
    for algo in ["a2c", "ppo"]:
        rows = [r for r in summary if r["algo"] == algo and not r["nan_in_val"] and r["max_val_trades"] > 0]
        if rows:
            winner = max(rows, key=lambda r: r["val_best_total_return"]
                         if not math.isnan(r["val_best_total_return"]) else float("-inf"))
            print(f"\n{algo.upper()} winner: lr={winner['lr']}  "
                  f"val_ret={winner['val_best_total_return']:+.6f}  "
                  f"val_eq=${winner['val_best_final_equity']:.2f}  "
                  f"trades={winner['val_best_trades']}")
        else:
            print(f"\n{algo.upper()}: no viable lr found")


if __name__ == "__main__":
    main()
