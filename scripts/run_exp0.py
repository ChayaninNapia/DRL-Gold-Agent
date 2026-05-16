"""Experiment 0: Reward Selection (PROPOSAL.md Sec. 5).

3 rewards (R1/R2/R4) x 3 algos (DDQN/A2C/PPO) x 3 seeds = 27 runs, no HPO.
Fixed hyperparameters from config.yaml (DDQN: prior best; A2C/PPO: literature
defaults). Reward normalization ON for all variants (PROPOSAL Sec. 6.4) so the
three are comparable at a single learning rate.

Winner = best **mean rank across the 3 algos on validation** for the configured
`train.best_metric` (default sortino) -- robust to algo idiosyncrasy. For each
(reward, algo) we mean over seeds; for each algo we rank rewards 1..3; mean
those ranks across algos -> reward with lowest mean rank wins.

Test metrics are reported per-run for transparency but are NEVER used for
selection (PROPOSAL Sec. 7).

Outputs (under runs/exp0/):
  - <reward>_<algo>_s<seed>/        one folder per run (full train+test artifacts)
  - exp0_summary.csv                one row per run with val + test metrics
  - exp0_winner.json                ranking table + declared winner
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

REWARDS = ["r1", "r2", "r4"]
ALGOS = ["ddqn", "a2c", "ppo"]
SEEDS = [42, 1337, 2026]

METRICS = [
    "total_return", "mdd", "trades", "winrate",
    "sharpe", "sortino", "avg_trade_pnl", "turnover", "avg_holding_time",
    "final_equity", "max_dd_dollar", "max_dd_pct", "ruin_rate",
]


_ALGO_CFG_SECTION = {"ddqn": "dqn", "a2c": "a2c", "ppo": "ppo"}


def _apply_per_reward_lr(cfg: dict, algo: str, reward_mode: str) -> None:
    """Apply algo[lr_per_reward][reward_mode] override to algo[lr] if present.

    Added 2026-05-16: R1 raw scale (~1e-4/bar) and R2/R4 raw scale (~$1-30/bar)
    differ by ~10^5 with normalize=false, so a single lr cannot work for all 3.
    Per-reward overrides in config.yaml encode the lrs found by quick A/B tests.
    Operates in-place on cfg.
    """
    section_name = _ALGO_CFG_SECTION[algo]
    section = cfg[section_name]
    override_map = section.get("lr_per_reward")
    if not override_map:
        return
    if reward_mode in override_map:
        section["lr"] = float(override_map[reward_mode])


def run_one(algo: str, cfg: dict, reward_mode: str, seed: int, run_name: str) -> dict:
    """Train one (algo, reward, seed) and return {val_metric, test_metrics}.
    `val_metric` is the best inner-val score for the configured best_metric."""
    cfg = copy.deepcopy(cfg)
    cfg["train"]["seed"] = seed
    cfg["run"]["run_name"] = run_name
    cfg["env"]["reward"]["mode"] = reward_mode
    _apply_per_reward_lr(cfg, algo, reward_mode)

    if algo == "ddqn":
        from train import train_ddqn
        test_m = train_ddqn(cfg)
    elif algo == "a2c":
        from train_a2c import train_a2c
        test_m = train_a2c(cfg)
    elif algo == "ppo":
        from train_ppo import train_ppo
        test_m = train_ppo(cfg)
    else:
        raise ValueError(f"Unknown algo: {algo!r}")

    # Read val best-checkpoint score for ranking. Trainer writes best_info.json
    # with val_<best_metric>. If no checkpoint was saved (all -inf), use NaN.
    run_dir = (WORKSPACE_ROOT / cfg["run"]["output_root"] / run_name).resolve()
    bi_path = run_dir / "best_info.json"
    if bi_path.exists():
        bi = json.loads(bi_path.read_text(encoding="utf-8"))
        metric_name = bi.get("metric", cfg["train"].get("best_metric", "sortino"))
        val_metric_val = float(bi.get(f"val_{metric_name}", float("nan")))
        val_trades = int(bi.get("val_trades", 0))
    else:
        metric_name = str(cfg["train"].get("best_metric", "sortino"))
        val_metric_val = float("nan")
        val_trades = 0

    return {
        "val_metric_name": metric_name,
        "val_metric": val_metric_val,
        "val_trades": val_trades,
        "test_metrics": test_m,
    }


def rank_rewards(rows: list[dict], ranking_metric: str) -> dict:
    """Compute mean rank across algos for each reward (lower rank = better).

    For each algo we mean the `ranking_metric` across seeds, then rank rewards
    1..3 (1 = best). The winner is the reward with the lowest mean rank across
    the 3 algos. NaNs are sorted to the bottom (worst rank).
    """
    # Build (reward, algo) -> mean across seeds
    mean_table: dict[tuple[str, str], float] = {}
    for reward in REWARDS:
        for algo in ALGOS:
            vals = [r["val_metric"] for r in rows
                    if r["reward"] == reward and r["algo"] == algo
                    and not np.isnan(r["val_metric"])]
            mean_table[(reward, algo)] = float(np.mean(vals)) if vals else float("nan")

    # Per-algo ranks
    rank_table: dict[tuple[str, str], int] = {}
    for algo in ALGOS:
        algo_scores = [(reward, mean_table[(reward, algo)]) for reward in REWARDS]
        # Sort descending (higher better); NaN sorts to the end.
        algo_scores.sort(key=lambda x: (float("-inf") if np.isnan(x[1]) else -x[1]))
        for rank, (reward, _) in enumerate(algo_scores, start=1):
            rank_table[(reward, algo)] = rank

    # Mean rank per reward
    reward_mean_rank = {}
    for reward in REWARDS:
        ranks = [rank_table[(reward, algo)] for algo in ALGOS]
        reward_mean_rank[reward] = float(np.mean(ranks))

    # Winner = lowest mean rank (ties broken by best raw mean across all algos)
    def tiebreak(reward: str) -> float:
        vals = [mean_table[(reward, algo)] for algo in ALGOS if not np.isnan(mean_table[(reward, algo)])]
        return -float(np.mean(vals)) if vals else float("inf")

    winner = min(REWARDS, key=lambda r: (reward_mean_rank[r], tiebreak(r)))

    return {
        "ranking_metric": ranking_metric,
        "mean_table": {f"{r}/{a}": v for (r, a), v in mean_table.items()},
        "rank_table": {f"{r}/{a}": v for (r, a), v in rank_table.items()},
        "reward_mean_rank": reward_mean_rank,
        "winner": winner,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 0: Reward Selection (3x3x3 = 27 runs)")
    parser.add_argument("--rewards", nargs="+", default=REWARDS, choices=REWARDS,
                        help="Reward modes to evaluate (default: r1 r2 r4)")
    parser.add_argument("--algos", nargs="+", default=ALGOS, choices=ALGOS,
                        help="Algorithms to evaluate (default: ddqn a2c ppo)")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                        help=f"Random seeds (default: {SEEDS})")
    parser.add_argument("--config", default=str(WORKSPACE_ROOT / "config.yaml"),
                        help="Path to base config YAML (default: config.yaml)")
    parser.add_argument("--run-prefix", default="exp0",
                        help="Run-name prefix and output subfolder (default: exp0)")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Override train.total_timesteps for all runs (default: from config)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip runs whose summary.json already exists (resume)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.total_timesteps is not None:
        cfg["train"]["total_timesteps"] = int(args.total_timesteps)
    # All Exp-0 runs share the prefix subfolder.
    cfg["run"]["output_root"] = str(Path(cfg["run"]["output_root"]) / args.run_prefix).replace("\\", "/")
    out_root = (WORKSPACE_ROOT / cfg["run"]["output_root"]).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    ranking_metric = str(cfg["train"].get("best_metric", "sortino"))

    print(f"Experiment 0 — Reward Selection")
    print(f"  Rewards:        {args.rewards}")
    print(f"  Algos:          {args.algos}")
    print(f"  Seeds:          {args.seeds}")
    print(f"  Total runs:     {len(args.rewards) * len(args.algos) * len(args.seeds)}")
    print(f"  Ranking metric: val_{ranking_metric}")
    print(f"  Total timesteps/run: {cfg['train']['total_timesteps']}")
    print(f"  Output root:    {out_root}")
    print()

    rows: list[dict] = []
    t_start = time.time()
    run_idx = 0
    total_runs = len(args.rewards) * len(args.algos) * len(args.seeds)

    for reward in args.rewards:
        for algo in args.algos:
            for seed in args.seeds:
                run_idx += 1
                run_name = f"{reward}_{algo}_s{seed}"
                run_dir = out_root / run_name
                summary_path = run_dir / "summary.json"

                print(f"\n{'='*70}")
                print(f"[{run_idx}/{total_runs}] reward={reward}  algo={algo.upper()}  seed={seed}  run={run_name}")
                print(f"{'='*70}")

                if args.skip_existing and summary_path.exists():
                    print(f"  -> skipping (summary.json exists)")
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    test_m = summary.get("test_metrics", {})
                    bi_path = run_dir / "best_info.json"
                    if bi_path.exists():
                        bi = json.loads(bi_path.read_text(encoding="utf-8"))
                        metric_name = bi.get("metric", ranking_metric)
                        val_metric_val = float(bi.get(f"val_{metric_name}", float("nan")))
                        val_trades = int(bi.get("val_trades", 0))
                    else:
                        val_metric_val = float("nan")
                        val_trades = 0
                    out = {
                        "val_metric_name": ranking_metric,
                        "val_metric": val_metric_val,
                        "val_trades": val_trades,
                        "test_metrics": test_m,
                    }
                else:
                    t_run = time.time()
                    out = run_one(algo, cfg, reward, seed, run_name)
                    print(f"  ... done in {time.time() - t_run:.1f}s")

                row = {
                    "reward": reward,
                    "algo": algo,
                    "seed": seed,
                    "run_name": run_name,
                    f"val_{ranking_metric}": out["val_metric"],
                    "val_trades": out["val_trades"],
                }
                for k in METRICS:
                    row[f"test_{k}"] = float(out["test_metrics"].get(k, float("nan")))
                rows.append(row)

    total_elapsed = time.time() - t_start
    print(f"\n\nAll {run_idx}/{total_runs} runs done in {total_elapsed/60:.1f} min")

    # --- write summary CSV ---
    summary_csv = out_root / "exp0_summary.csv"
    fieldnames = ["reward", "algo", "seed", "run_name", f"val_{ranking_metric}", "val_trades"] \
        + [f"test_{k}" for k in METRICS]
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Per-run summary -> {summary_csv}")

    # --- rank rewards across algos ---
    # Use the rows we just collected (with normalized field name) so we don't
    # have to re-read JSON files. Adapt the format expected by rank_rewards().
    rank_rows = [
        {
            "reward": r["reward"],
            "algo": r["algo"],
            "seed": r["seed"],
            "val_metric": r[f"val_{ranking_metric}"],
        }
        for r in rows
    ]
    ranking = rank_rewards(rank_rows, ranking_metric)

    winner_path = out_root / "exp0_winner.json"
    winner_path.write_text(json.dumps({
        "ranking_metric": ranking_metric,
        "rewards_tested": args.rewards,
        "algos_tested": args.algos,
        "seeds_tested": args.seeds,
        **ranking,
        "total_minutes": total_elapsed / 60.0,
    }, indent=2), encoding="utf-8")
    print(f"Winner JSON     -> {winner_path}")

    # --- console report ---
    print(f"\n{'-'*70}")
    print(f"Exp-0 Ranking by val_{ranking_metric} (mean over seeds, then ranked per algo)")
    print(f"{'-'*70}")
    print(f"{'reward':<6} | " + " | ".join(f"{a:>14}" for a in args.algos) + " | mean rank")
    print(f"{'-'*70}")
    for reward in args.rewards:
        cells = []
        for algo in args.algos:
            mean = ranking["mean_table"].get(f"{reward}/{algo}", float("nan"))
            rank = ranking["rank_table"].get(f"{reward}/{algo}", "?")
            cells.append(f"{mean:>+8.3f} (#{rank})")
        mean_rank = ranking["reward_mean_rank"][reward]
        print(f"{reward:<6} | " + " | ".join(cells) + f" | {mean_rank:>6.2f}")
    print(f"{'-'*70}")
    print(f"\nWINNER: {ranking['winner']}  (lowest mean rank)")
    print(f"\nNext step: re-run Experiment 1 (algo comparison + full HPO) with "
          f"env.reward.mode = {ranking['winner']!r}.")


if __name__ == "__main__":
    main()
