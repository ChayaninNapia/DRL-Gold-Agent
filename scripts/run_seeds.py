"""Multi-seed training runner.

Runs DDQN, A2C, or PPO (or all three) over a list of seeds, then aggregates
test metrics across seeds into:
  - <output_root>/<base_run_name>/seeds_summary.csv   — one row per seed
  - <output_root>/<base_run_name>/seeds_aggregate.json — mean ± std across seeds

Usage (from workspace root):
    python scripts/run_seeds.py --algo ddqn
    python scripts/run_seeds.py --algo a2c --seeds 42 1337 2026
    python scripts/run_seeds.py --algo ppo --seeds 42 1337 2026 --run-name ppo_exp1
    python scripts/run_seeds.py --algo all --seeds 42 1337 2026

Each seed's run is saved under:
    <output_root>/<base_run_name>_s<seed>/

The base_run_name defaults to config.yaml run.run_name; override with --run-name.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

METRICS = [
    "total_return", "mdd", "trades", "winrate",
    "sharpe", "sortino", "avg_trade_pnl", "turnover", "avg_holding_time",
    "final_equity", "max_dd_dollar", "max_dd_pct", "ruin_rate",
]


def run_one(algo: str, cfg: dict, seed: int, run_name: str) -> dict:
    """Train one algorithm with the given seed and return test metrics."""
    cfg = copy.deepcopy(cfg)
    cfg["train"]["seed"] = seed
    cfg["run"]["run_name"] = run_name

    if algo == "ddqn":
        from train import train_ddqn
        return train_ddqn(cfg)
    elif algo == "a2c":
        from train_a2c import train_a2c
        return train_a2c(cfg)
    elif algo == "ppo":
        from train_ppo import train_ppo
        return train_ppo(cfg)
    else:
        raise ValueError(f"Unknown algo: {algo!r}")


def aggregate(rows: list[dict]) -> dict:
    """Compute mean and std for each metric across seeds."""
    agg: dict = {}
    for key in METRICS:
        vals = [float(r[key]) for r in rows if key in r]
        if vals:
            agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed DRL training runner")
    parser.add_argument("--algo", required=True, choices=["ddqn", "a2c", "ppo", "all"],
                        help="Algorithm to train. 'all' runs ddqn, a2c, ppo in sequence.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2026],
                        help="Random seeds to run (default: 42 1337 2026)")
    parser.add_argument("--run-name", default=None,
                        help="Base run name (default: config.yaml run.run_name)")
    parser.add_argument("--config", default=str(WORKSPACE_ROOT / "config.yaml"),
                        help="Path to config YAML (default: config.yaml)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base_name = args.run_name or str(cfg["run"]["run_name"])
    output_root = str(cfg["run"]["output_root"])

    algos = ["ddqn", "a2c", "ppo"] if args.algo == "all" else [args.algo]

    for algo in algos:
        print(f"\n{'='*60}")
        print(f"Algorithm: {algo.upper()}  Seeds: {args.seeds}")
        print(f"{'='*60}")

        rows: list[dict] = []
        run_names: list[str] = []

        for seed in args.seeds:
            run_name = f"{base_name}_{algo}_s{seed}"
            run_names.append(run_name)
            print(f"\n--- {algo.upper()} seed={seed}  run={run_name} ---")
            test_m = run_one(algo, cfg, seed, run_name)
            row = {"algo": algo, "seed": seed, "run_name": run_name}
            row.update({k: test_m.get(k, float("nan")) for k in METRICS})
            rows.append(row)
            print(f"    test: ret={test_m['total_return']:+.4f}  "
                  f"sharpe={test_m['sharpe']:+.3f}  sortino={test_m['sortino']:+.3f}  "
                  f"mdd={test_m['mdd']:+.4f}  trades={test_m['trades']}  "
                  f"winrate={test_m['winrate']:.3f}")

        # --- output directory: <output_root>/<base_name>_<algo>/ ---
        out_dir = (WORKSPACE_ROOT / output_root / f"{base_name}_{algo}").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # seeds_summary.csv
        summary_path = out_dir / "seeds_summary.csv"
        fieldnames = ["algo", "seed", "run_name"] + METRICS
        with summary_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[{algo.upper()}] Summary written -> {summary_path}")

        # seeds_aggregate.json
        agg = aggregate(rows)
        agg_payload = {
            "algo": algo,
            "base_run_name": base_name,
            "seeds": args.seeds,
            "run_names": run_names,
            "test_metrics": agg,
        }
        agg_path = out_dir / "seeds_aggregate.json"
        agg_path.write_text(json.dumps(agg_payload, indent=2), encoding="utf-8")
        print(f"[{algo.upper()}] Aggregate  written -> {agg_path}")

        # Print summary table
        print(f"\n[{algo.upper()}] Results across seeds:")
        print(f"  {'seed':>6}  {'ret':>8}  {'sharpe':>8}  {'sortino':>8}  {'mdd':>8}  {'trades':>7}  {'winrate':>7}")
        for r in rows:
            print(f"  {r['seed']:>6}  {r['total_return']:>+8.4f}  "
                  f"{r['sharpe']:>+8.3f}  {r['sortino']:>+8.3f}  "
                  f"{r['mdd']:>+8.4f}  {int(r['trades']):>7}  {r['winrate']:>7.3f}")
        print(f"  {'mean':>6}  {agg['total_return']['mean']:>+8.4f}  "
              f"{agg['sharpe']['mean']:>+8.3f}  {agg['sortino']['mean']:>+8.3f}  "
              f"{agg['mdd']['mean']:>+8.4f}  {agg['trades']['mean']:>7.1f}  "
              f"{agg['winrate']['mean']:>7.3f}")
        print(f"  {'std':>6}  {agg['total_return']['std']:>8.4f}  "
              f"{agg['sharpe']['std']:>8.3f}  {agg['sortino']['std']:>8.3f}  "
              f"{agg['mdd']['std']:>8.4f}  {agg['trades']['std']:>7.1f}  "
              f"{agg['winrate']['std']:>7.3f}")


if __name__ == "__main__":
    main()
