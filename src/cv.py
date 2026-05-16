"""Time-series cross-validation utilities for hyperparameter optimization.

Inner CV is applied on the train split only (held-out val/test never touched here).
Layout: expanding-window with equal val size per fold (Option A).

For train of N days and k folds with val_size v per fold:
  fold i (1..k): inner_train = days[0 : N - (k-i+1)*v]
                 inner_val   = days[N - (k-i+1)*v : N - (k-i)*v]

Default for this project: N=600 train days, k=5 folds, v=24 → folds end at
[504, 528, 552, 576, 600]; inner-train start sizes [480, 504, 528, 552, 576].
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class CVFold:
    index: int
    inner_train_dates: list[pd.Timestamp]
    inner_val_dates: list[pd.Timestamp]


def expanding_window_folds(
    train_dates: list[pd.Timestamp],
    n_folds: int = 5,
    val_size: int = 24,
) -> list[CVFold]:
    """Generate expanding-window CV folds from the train date list.

    All folds use the same `val_size`. Inner-train grows by `val_size` each fold.
    The final fold's inner-val ends exactly at the last train date.
    """
    n = len(train_dates)
    required = n_folds * val_size
    assert n >= required + val_size, (
        f"need at least {required + val_size} train days for {n_folds} folds "
        f"of val_size={val_size} (got {n}); first fold's inner-train would be empty"
    )
    folds: list[CVFold] = []
    for i in range(n_folds):
        val_end = n - (n_folds - 1 - i) * val_size
        val_start = val_end - val_size
        folds.append(
            CVFold(
                index=i + 1,
                inner_train_dates=train_dates[:val_start],
                inner_val_dates=train_dates[val_start:val_end],
            )
        )
    return folds


def aggregate_fold_scores(scores: list[float], penalty: float = 0.5) -> float:
    """Combine per-fold scores into a single trial objective.

    Returns `mean(scores) - penalty * std(scores)`. Penalizes unstable
    hyperparameter configs — rewards consistency across regimes.
    """
    import statistics as _s

    if len(scores) == 0:
        return float("-inf")
    if len(scores) == 1:
        return scores[0]
    mu = _s.fmean(scores)
    sigma = _s.pstdev(scores)
    return mu - penalty * sigma


if __name__ == "__main__":
    import yaml
    from pathlib import Path

    from data import load_raw, select_window, split_days

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "config.yaml").read_text(encoding="utf-8")
    )
    df = load_raw(cfg["data"]["path"])
    df = select_window(df, cfg["data"]["window_days"])
    split = split_days(df, cfg["data"]["n_train"], cfg["data"]["n_val"], cfg["data"]["n_test"])

    folds = expanding_window_folds(
        split.train_dates,
        n_folds=cfg["cv"]["n_folds"],
        val_size=cfg["cv"]["val_size"],
    )
    print(f"Generated {len(folds)} folds from {len(split.train_dates)} train days")
    for f in folds:
        print(
            f"  fold {f.index}: inner_train={len(f.inner_train_dates):3d} days "
            f"[{f.inner_train_dates[0].date()} -> {f.inner_train_dates[-1].date()}]  "
            f"inner_val={len(f.inner_val_dates):2d} days "
            f"[{f.inner_val_dates[0].date()} -> {f.inner_val_dates[-1].date()}]"
        )
