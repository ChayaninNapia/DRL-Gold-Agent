"""Baseline strategy evaluation runner.

Evaluates all 5 baselines on the test split (and optionally val) then writes:
  - <output_root>/baselines/baselines_results.csv  — one row per baseline
  - <output_root>/baselines/baselines_results.json — same data as JSON

Usage (from workspace root):
    python scripts/run_baselines.py
    python scripts/run_baselines.py --split test          # default
    python scripts/run_baselines.py --split val
    python scripts/run_baselines.py --split both
    python scripts/run_baselines.py --ma-fast 10 --ma-slow 40

Baselines evaluated:
  flat            — always flat (zero-trade)
  long            — enter long at session open, hold to EOD
  short           — enter short at session open, hold to EOD
  random          — uniformly random action each bar (seed from config)
  ma_crossover    — long when fast MA > slow MA, short otherwise
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

METRICS = [
    "total_return", "mdd", "trades", "winrate",
    "sharpe", "sortino", "avg_trade_pnl", "turnover", "avg_holding_time",
    "final_equity", "max_dd_dollar", "max_dd_pct", "ruin_rate",
]


def evaluate_baseline(agent, feat, dates, cfg, phase: str, trade_log=None) -> dict:
    from train import evaluate_policy_per_session
    return evaluate_policy_per_session(agent, feat, dates, cfg, trade_log=trade_log, phase=phase)


def run_baselines(cfg: dict, split, feat, splits_to_run: list[str],
                  ma_fast: int, ma_slow: int, seed: int, out_dir) -> list[dict]:
    import csv as _csv

    from baselines import FlatBaseline, LongBaseline, ShortBaseline, RandomBaseline, MACrossoverBaseline

    agents = [
        FlatBaseline(cfg),
        LongBaseline(cfg),
        ShortBaseline(cfg),
        RandomBaseline(cfg, seed=seed),
        MACrossoverBaseline(cfg, fast=ma_fast, slow=ma_slow),
    ]

    rows: list[dict] = []
    for phase in splits_to_run:
        dates = split.val_dates if phase == "val" else split.test_dates
        print(f"\n--- Evaluating baselines on {phase} ({len(dates)} sessions) ---")
        for agent in agents:
            trades: list[dict] = []
            m = evaluate_baseline(agent, feat, dates, cfg, phase=phase, trade_log=trades)
            row = {"baseline": agent.name, "phase": phase}
            row.update({k: m[k] for k in METRICS})
            rows.append(row)

            # Per-baseline trade log for visualization (same schema as the
            # DRL trainers' trades.csv: one row per completed trade).
            tpath = out_dir / f"{agent.name}_{phase}_trades.csv"
            with tpath.open("w", newline="", encoding="utf-8") as fh:
                w = _csv.writer(fh)
                w.writerow(["baseline", "phase", "day", "entry_time", "exit_time",
                            "side", "entry_price", "exit_price", "bars_held", "pnl_log"])
                for tr in trades:
                    w.writerow([agent.name, tr["phase"], tr["day"], tr["entry_time"],
                                tr["exit_time"], tr["side"], tr["entry_price"],
                                tr["exit_price"], tr["bars_held"], tr["pnl_log"]])

            print(f"  {agent.name:<16}  eq=${m['final_equity']:.2f}  "
                  f"ret={m['total_return']:+.4f}  sharpe={m['sharpe']:+.3f}  "
                  f"sortino={m['sortino']:+.3f}  mdd=${m['max_dd_dollar']:.0f}  "
                  f"trades={m['trades']}  winrate={m['winrate']:.3f}  "
                  f"ruin={m['ruin_rate']:.2f}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline strategy evaluation")
    parser.add_argument("--split", default="test", choices=["test", "val", "both"],
                        help="Which split(s) to evaluate on (default: test)")
    parser.add_argument("--ma-fast", type=int, default=20,
                        help="MA crossover fast window in bars (default: 20)")
    parser.add_argument("--ma-slow", type=int, default=60,
                        help="MA crossover slow window in bars (default: 60)")
    parser.add_argument("--config", default=str(WORKSPACE_ROOT / "config.yaml"),
                        help="Path to config YAML (default: config.yaml)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg["train"]["seed"])

    from data import load_raw, select_window, split_days
    from features import build_features

    df = load_raw(cfg["data"]["path"])
    df = select_window(df, cfg["data"]["window_days"])
    feat = build_features(df, cfg["features"])
    split = split_days(feat, cfg["data"]["n_train"], cfg["data"]["n_val"], cfg["data"]["n_test"])
    print(f"Sessions  train/val/test = {len(split.train_dates)}/{len(split.val_dates)}/{len(split.test_dates)}")
    print(f"Action space: {cfg['env']['action_space']}  capital=${cfg['env'].get('capital', 10000):.0f}  "
          f"lot={cfg['env'].get('lot', 0.01)}  reward.mode={cfg['env'].get('reward', {}).get('mode', 'r1')}")

    splits_to_run = ["val", "test"] if args.split == "both" else [args.split]

    out_dir = (WORKSPACE_ROOT / cfg["run"]["output_root"] / "baselines").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = run_baselines(cfg, split, feat, splits_to_run,
                         ma_fast=args.ma_fast, ma_slow=args.ma_slow, seed=seed,
                         out_dir=out_dir)

    # --- write output ---

    csv_path = out_dir / "baselines_results.csv"
    fieldnames = ["baseline", "phase"] + METRICS
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults written -> {csv_path}")

    json_path = out_dir / "baselines_results.json"
    payload = {
        "config": {
            "action_space": cfg["env"]["action_space"],
            "capital": cfg["env"].get("capital", 10000.0),
            "lot": cfg["env"].get("lot", 0.01),
            "reward_mode": cfg["env"].get("reward", {}).get("mode", "r1"),
            "ma_fast": args.ma_fast,
            "ma_slow": args.ma_slow,
            "seed": seed,
            "window_days": cfg["data"]["window_days"],
        },
        "results": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Results written -> {json_path}")

    # --- summary table ---
    print(f"\n{'Baseline':<16}  {'Phase':<5}  {'Equity':>9}  {'Return':>8}  "
          f"{'Sharpe':>8}  {'Sortino':>8}  {'MDD $':>8}  {'Trades':>7}  {'Ruin':>5}")
    print("-" * 95)
    for r in rows:
        print(f"{r['baseline']:<16}  {r['phase']:<5}  "
              f"${r['final_equity']:>8.2f}  {r['total_return']:>+8.4f}  "
              f"{r['sharpe']:>+8.3f}  {r['sortino']:>+8.3f}  "
              f"${r['max_dd_dollar']:>6.0f}  {int(r['trades']):>7}  "
              f"{r['ruin_rate']:>5.2f}")


if __name__ == "__main__":
    main()
