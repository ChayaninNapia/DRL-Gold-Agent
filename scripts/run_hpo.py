"""Hyperparameter optimization entry point (Aggressive plan, JOURNAL.md 2026-05-15).

Runs Optuna TPE search with a Hyperband pruner over an inner expanding-window
CV on the train split. Held-out val/test are never touched here.

Usage (from workspace root):
    python scripts/run_hpo.py --algo ddqn
    python scripts/run_hpo.py --algo a2c --n-trials 12 --n-folds 3 --hpo-timesteps 100000
    python scripts/run_hpo.py --algo ppo --study-name ppo_aggressive

Defaults match the "Aggressive" Experiment-1 plan: 12 trials, 3 folds,
100k timesteps/fold, TPESampler + HyperbandPruner. Storage is a sqlite db so
the study can be resumed (re-running the same --study-name continues it).

Output: runs/hpo/<algo>/<study-name>_best.json (best trial params + value).
Per-trial/per-fold training artifacts land in runs/hpo_<study>_t###_f#/ and
TB logs in runs/_tb/hpo_<study>_t###_f#/ (consistent with the train adapter).

Console output is ASCII-only (cp874 stdout on the Thai-locale system Python).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

import optuna

from cv import expanding_window_folds
from hpo import make_objective


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna HPO with inner expanding-window CV")
    parser.add_argument("--algo", required=True, choices=["ddqn", "a2c", "ppo"])
    parser.add_argument("--n-trials", type=int, default=12,
                        help="Optuna trials (Aggressive plan default: 12)")
    parser.add_argument("--n-folds", type=int, default=3,
                        help="Inner-CV folds (Aggressive plan default: 3)")
    parser.add_argument("--val-size", type=int, default=None,
                        help="Days per inner-val fold (default: cfg['cv']['val_size'])")
    parser.add_argument("--hpo-timesteps", type=int, default=100000,
                        help="Timesteps per fold (Aggressive plan default: 100000)")
    parser.add_argument("--study-name", default=None,
                        help="Optuna study name (default: <algo>_aggressive)")
    parser.add_argument("--config", default=str(WORKSPACE_ROOT / "config.yaml"))
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    algo = args.algo
    study_name = args.study_name or f"{algo}_aggressive"
    val_size = args.val_size if args.val_size is not None else int(cfg["cv"]["val_size"])

    from data import load_raw, select_window, split_days
    from features import build_features

    df = load_raw(cfg["data"]["path"])
    df = select_window(df, cfg["data"]["window_days"])
    feat = build_features(df, cfg["features"])
    split = split_days(feat, cfg["data"]["n_train"], cfg["data"]["n_val"], cfg["data"]["n_test"])

    folds = expanding_window_folds(split.train_dates, n_folds=args.n_folds, val_size=val_size)
    print(f"Algo={algo}  study={study_name}  trials={args.n_trials}  "
          f"folds={args.n_folds}x{val_size}d  timesteps/fold={args.hpo_timesteps}")
    for f in folds:
        print(f"  fold {f.index}: inner_train={len(f.inner_train_dates)}d "
              f"[{f.inner_train_dates[0].date()} -> {f.inner_train_dates[-1].date()}]  "
              f"inner_val={len(f.inner_val_dates)}d "
              f"[{f.inner_val_dates[0].date()} -> {f.inner_val_dates[-1].date()}]")

    out_dir = (WORKSPACE_ROOT / cfg["run"]["output_root"] / "hpo" / algo).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(out_dir / f'{study_name}.db').as_posix()}"

    sampler = optuna.samplers.TPESampler(seed=int(cfg["train"]["seed"]))
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=1, max_resource=args.n_folds, reduction_factor=3
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    objective = make_objective(
        algo=algo,
        base_cfg=cfg,
        folds=folds,
        hpo_timesteps=args.hpo_timesteps,
        study_name=study_name,
    )

    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    best = study.best_trial
    payload = {
        "algo": algo,
        "study_name": study_name,
        "storage": storage,
        "n_trials_run": len(study.trials),
        "n_folds": args.n_folds,
        "val_size": val_size,
        "hpo_timesteps": args.hpo_timesteps,
        "objective_metric": str(cfg["cv"].get("objective_metric", "sortino")),
        "best_trial": {
            "number": best.number,
            "value": best.value,
            "params": best.params,
            "fold_scores": best.user_attrs.get("fold_scores"),
            "fold_trades": best.user_attrs.get("fold_trades"),
            "mean_val_trades": best.user_attrs.get("mean_val_trades"),
        },
    }
    best_path = out_dir / f"{study_name}_best.json"
    best_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n[{algo.upper()}] HPO done. trials={len(study.trials)}  "
          f"best_value={best.value:+.6f}  best_trial=#{best.number}")
    print(f"[{algo.upper()}] best params: {best.params}")
    print(f"[{algo.upper()}] best-trial JSON -> {best_path}")


if __name__ == "__main__":
    main()
