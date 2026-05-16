"""Optuna hyperparameter optimization with inner expanding-window CV.

Operates on the train split only. The held-out validation/test splits from
split_days() are never touched here (the train_*() adapter early-returns before
the test rollout when cfg["_hpo"] is set).

One trial = one hyperparameter configuration, evaluated over all inner-CV folds.
Per-fold score = best inner-val <objective_metric> reached during that fold's
training run (the train adapter returns this). Trial objective =
aggregate_fold_scores(per-fold scores) = mean - agg_penalty * std.

A trial whose best inner-val checkpoint trades fewer than cfg["cv"]["min_trades"]
times (averaged across folds) is disqualified by returning -inf — this filters
degenerate "do-nothing" policies that score deceptively well on near-zero return
sequences (see JOURNAL.md 2026-05-14 normalization ablation).

Pruning: trial.report(fold_score, fold_index) after each fold; trial.should_prune()
lets the study's pruner (Hyperband) cut unpromising trials early.

Per-algo search spaces follow the 2026-05-15 "Aggressive" plan in JOURNAL.md.
"""
from __future__ import annotations

import copy
from typing import Callable

import optuna
import pandas as pd

from cv import CVFold, aggregate_fold_scores

# Trial dispatch is lazy-imported inside _run_fold so importing hpo.py does not
# pull in torch before the caller wants it.

# The DDQN config section is named "dqn" (historical), not "ddqn".
_CFG_SECTION = {"ddqn": "dqn", "a2c": "a2c", "ppo": "ppo"}


def _sample_hparams(trial: optuna.Trial, algo: str, reward_mode: str = "r1") -> dict:
    """Per-algo search space (2026-05-15 Aggressive plan, JOURNAL.md).

    When `reward_mode == "r4"` the R4 hyperparameters (`reward_beta`,
    `reward_dd_thresh`) are also sampled — PROPOSAL Sec. 3.6: they are tuned
    only in Exp 1 for the Exp-0 winning reward, not in Exp 0 itself.
    The "reward_" prefix on these keys triggers special placement in
    `_apply_hparams` (they go into cfg["env"]["reward"], not the algo section).
    """
    hidden_choices = {"64x64": [64, 64], "128x128": [128, 128], "256x256": [256, 256]}
    hkey = trial.suggest_categorical("hidden_sizes", list(hidden_choices.keys()))
    hidden = hidden_choices[hkey]

    if algo == "ddqn":
        hp = {
            "hidden_sizes": hidden,
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "target_update_interval": trial.suggest_categorical(
                "target_update_interval", [500, 1000, 2000]
            ),
            "exploration_fraction": trial.suggest_float("exploration_fraction", 0.3, 0.8),
        }
    elif algo == "a2c":
        hp = {
            "hidden_sizes": hidden,
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "entropy_coef": trial.suggest_float("entropy_coef", 1e-3, 0.1, log=True),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
        }
    elif algo == "ppo":
        hp = {
            "hidden_sizes": hidden,
            "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
            "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
            "n_epochs": trial.suggest_categorical("n_epochs", [5, 10, 20]),
            "minibatch_size": trial.suggest_categorical("minibatch_size", [64, 128, 256]),
            "entropy_coef": trial.suggest_float("entropy_coef", 1e-3, 0.1, log=True),
        }
    else:
        raise ValueError(f"Unknown algo: {algo!r}")

    # R4-specific search (only when running HPO on the R4 reward).
    if reward_mode == "r4":
        hp["reward_beta"] = trial.suggest_float("reward_beta", 0.1, 10.0, log=True)
        hp["reward_dd_thresh"] = trial.suggest_float("reward_dd_thresh", 0.005, 0.10, log=True)

    return hp


def _apply_hparams(cfg: dict, algo: str, hp: dict) -> dict:
    """Return a deep-copied cfg with sampled hyperparameters written in.

    Keys with the "reward_" prefix go into cfg["env"]["reward"] (R4 tuning
    knobs `beta`, `dd_thresh`); all others go into the algo's config section.
    """
    cfg = copy.deepcopy(cfg)
    section = cfg[_CFG_SECTION[algo]]
    cfg["env"].setdefault("reward", {})
    for k, v in hp.items():
        if k.startswith("reward_"):
            cfg["env"]["reward"][k[len("reward_"):]] = v
        else:
            section[k] = v
    return cfg


def _run_fold(
    algo: str,
    cfg: dict,
    fold: CVFold,
    timesteps: int,
    run_name: str,
) -> dict:
    """Train one algo on one CV fold via the train_*() HPO adapter.

    Returns the adapter's HPO dict: {hpo_objective, best_metric, val_trades}.
    """
    if algo == "ddqn":
        from train import train_ddqn as train_fn
    elif algo == "a2c":
        from train_a2c import train_a2c as train_fn
    elif algo == "ppo":
        from train_ppo import train_ppo as train_fn
    else:
        raise ValueError(f"Unknown algo: {algo!r}")

    cfg = copy.deepcopy(cfg)
    cfg["run"]["run_name"] = run_name
    cfg["_hpo"] = {
        "inner_train_dates": fold.inner_train_dates,
        "inner_val_dates": fold.inner_val_dates,
        "timesteps_override": int(timesteps),
    }
    return train_fn(cfg)


def make_objective(
    algo: str,
    base_cfg: dict,
    folds: list[CVFold],
    hpo_timesteps: int,
    study_name: str,
) -> Callable[[optuna.Trial], float]:
    """Build the Optuna objective closure.

    `base_cfg`     — the loaded config.yaml dict (not mutated).
    `folds`        — inner-CV folds from cv.expanding_window_folds().
    `hpo_timesteps`— per-fold timestep budget (Aggressive plan: 100k).
    `study_name`   — used to namespace per-trial run dirs / TB logs.
    """
    cv_cfg = base_cfg.get("cv", {})
    agg_penalty = float(cv_cfg.get("agg_penalty", 0.5))
    min_trades = int(cv_cfg.get("min_trades", 50))
    objective_metric = str(cv_cfg.get("objective_metric", "sortino")).lower()

    reward_mode = str(base_cfg.get("env", {}).get("reward", {}).get("mode", "r1")).lower()

    def objective(trial: optuna.Trial) -> float:
        hp = _sample_hparams(trial, algo, reward_mode=reward_mode)
        cfg = _apply_hparams(base_cfg, algo, hp)
        # The objective metric drives best-checkpoint selection inside train_*().
        cfg["train"]["best_metric"] = objective_metric

        fold_scores: list[float] = []
        fold_trades: list[int] = []
        for fold in folds:
            run_name = f"hpo_{study_name}_t{trial.number:03d}_f{fold.index}"
            result = _run_fold(algo, cfg, fold, hpo_timesteps, run_name)
            score = float(result["hpo_objective"])
            fold_scores.append(score)
            fold_trades.append(int(result.get("val_trades", 0)))

            # Report the running aggregate so the pruner compares like-for-like
            # across trials at the same fold index.
            running = aggregate_fold_scores(fold_scores, penalty=agg_penalty)
            trial.report(running, fold.index)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_trades = sum(fold_trades) / len(fold_trades) if fold_trades else 0.0
        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("fold_trades", fold_trades)
        trial.set_user_attr("mean_val_trades", mean_trades)

        # Disqualify degenerate do-nothing policies.
        if mean_trades < min_trades:
            trial.set_user_attr("disqualified", f"mean_val_trades={mean_trades:.1f} < {min_trades}")
            return float("-inf")

        return aggregate_fold_scores(fold_scores, penalty=agg_penalty)

    return objective
