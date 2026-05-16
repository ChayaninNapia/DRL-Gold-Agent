"""Final retrain + test evaluation using the best HPO trial.

Pulls the best trial from a completed HPO study (sqlite db or the *_best.json
written by run_hpo.py), writes its hyperparameters into the algo's config
section, then retrains at the full production budget (500k timesteps) over
`--seeds` on the full 600-day train split. Held-out val(75) drives best-ckpt
selection and early stop; test(75) is evaluated exactly once per seed.

This is the only place the held-out test split is touched for an algorithm.

Usage (from workspace root):
    python scripts/run_final.py --algo ddqn
    python scripts/run_final.py --algo ppo --seeds 42 1337 2026 --study-name ppo_aggressive

Output: runs/final_<algo>/seeds_summary.csv + seeds_aggregate.json
        (same schema as run_seeds.py), plus per-seed run dirs runs/final_<algo>_s<seed>/.

Console output is ASCII-only (cp874 stdout on the Thai-locale system Python).
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

# Optuna stores hidden_sizes as a categorical key; map back to the list form
# the agents expect. Must stay in sync with hpo._sample_hparams.
_HIDDEN_MAP = {"64x64": [64, 64], "128x128": [128, 128], "256x256": [256, 256]}

# The DDQN config section is named "dqn" (historical), not "ddqn".
_CFG_SECTION = {"ddqn": "dqn", "a2c": "a2c", "ppo": "ppo"}


def _load_best_params(algo: str, study_name: str, cfg: dict) -> dict:
    """Read best-trial params from the run_hpo.py JSON (falls back to the
    sqlite study if the JSON is missing)."""
    out_dir = (WORKSPACE_ROOT / cfg["run"]["output_root"] / "hpo" / algo).resolve()
    json_path = out_dir / f"{study_name}_best.json"
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        return payload["best_trial"]["params"]

    import optuna
    storage = f"sqlite:///{(out_dir / f'{study_name}.db').as_posix()}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    return study.best_trial.params


def _apply_params(cfg: dict, algo: str, params: dict) -> dict:
    """Write tuned hyperparameters into cfg.

    Keys with the "reward_" prefix go into cfg["env"]["reward"] (matches
    hpo._apply_hparams). `hidden_sizes` is translated via _HIDDEN_MAP. All
    other keys go into the algo's config section.
    """
    cfg = copy.deepcopy(cfg)
    section = cfg[_CFG_SECTION[algo]]
    cfg["env"].setdefault("reward", {})
    for k, v in params.items():
        if k == "hidden_sizes":
            section[k] = _HIDDEN_MAP[v]
        elif k.startswith("reward_"):
            cfg["env"]["reward"][k[len("reward_"):]] = v
        else:
            section[k] = v
    return cfg


def _train(algo: str, cfg: dict):
    if algo == "ddqn":
        from train import train_ddqn as fn
    elif algo == "a2c":
        from train_a2c import train_a2c as fn
    elif algo == "ppo":
        from train_ppo import train_ppo as fn
    else:
        raise ValueError(f"Unknown algo: {algo!r}")
    return fn(cfg)


def aggregate(rows: list[dict]) -> dict:
    agg: dict = {}
    for key in METRICS:
        vals = [float(r[key]) for r in rows if key in r]
        if vals:
            agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Final retrain + test with best HPO params")
    parser.add_argument("--algo", required=True, choices=["ddqn", "a2c", "ppo"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2026])
    parser.add_argument("--study-name", default=None,
                        help="HPO study name (default: <algo>_aggressive)")
    parser.add_argument("--config", default=str(WORKSPACE_ROOT / "config.yaml"))
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    algo = args.algo
    study_name = args.study_name or f"{algo}_aggressive"

    params = _load_best_params(algo, study_name, cfg)
    print(f"[{algo.upper()}] best HPO params ({study_name}): {params}")

    tuned_cfg = _apply_params(cfg, algo, params)
    # Final retrain uses the production budget and the objective metric for
    # best-checkpoint selection (consistent with HPO).
    tuned_cfg["train"]["best_metric"] = str(cfg["cv"].get("objective_metric", "sortino")).lower()

    rows: list[dict] = []
    run_names: list[str] = []
    for seed in args.seeds:
        run_name = f"final_{algo}_s{seed}"
        run_names.append(run_name)
        c = copy.deepcopy(tuned_cfg)
        c["train"]["seed"] = seed
        c["run"]["run_name"] = run_name
        print(f"\n--- {algo.upper()} FINAL seed={seed}  run={run_name} ---")
        test_m = _train(algo, c)
        row = {"algo": algo, "seed": seed, "run_name": run_name}
        row.update({k: test_m.get(k, float("nan")) for k in METRICS})
        rows.append(row)
        print(f"    test: eq=${test_m['final_equity']:.2f}  ret={test_m['total_return']:+.4f}  "
              f"sharpe={test_m['sharpe']:+.3f}  sortino={test_m['sortino']:+.3f}  "
              f"mdd=${test_m['max_dd_dollar']:.0f}  trades={test_m['trades']}  "
              f"ruin={test_m['ruin_rate']:.2f}")

    out_dir = (WORKSPACE_ROOT / cfg["run"]["output_root"] / f"final_{algo}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "seeds_summary.csv"
    fieldnames = ["algo", "seed", "run_name"] + METRICS
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[{algo.upper()}] Summary written -> {summary_path}")

    agg = aggregate(rows)
    agg_payload = {
        "algo": algo,
        "study_name": study_name,
        "best_params": params,
        "seeds": args.seeds,
        "run_names": run_names,
        "test_metrics": agg,
    }
    agg_path = out_dir / "seeds_aggregate.json"
    agg_path.write_text(json.dumps(agg_payload, indent=2), encoding="utf-8")
    print(f"[{algo.upper()}] Aggregate  written -> {agg_path}")

    print(f"\n[{algo.upper()}] Test results across seeds:")
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
