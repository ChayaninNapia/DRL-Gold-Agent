"""Data loading and per-day splitting for M1 intraday DRL.

Workspace root: d:\\EA. Run with d:\\EA\\.venv\\Scripts\\python.exe.

1 episode = 1 trading day (all M1 bars available for that calendar date).
Episode length varies per day — keep whatever bars exist; do not pad or truncate.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class DaySplit:
    train_dates: list[pd.Timestamp]
    val_dates: list[pd.Timestamp]
    test_dates: list[pd.Timestamp]


def load_raw(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.is_absolute():
        p = WORKSPACE_ROOT / p
    df = pd.read_parquet(p)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    df["session_date"] = df["time"].dt.date
    return df


def select_window(df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """Keep only the last `window_days` distinct trading dates."""
    dates = sorted(df["session_date"].unique())
    keep = set(dates[-window_days:])
    return df.loc[df["session_date"].isin(keep)].reset_index(drop=True).copy()


def split_days(df: pd.DataFrame, n_train: int, n_val: int, n_test: int) -> DaySplit:
    dates = sorted(df["session_date"].unique())
    n = len(dates)
    assert n_train + n_val + n_test == n, (
        f"split sizes {n_train}+{n_val}+{n_test}={n_train + n_val + n_test} "
        f"must equal number of trading days ({n})"
    )
    return DaySplit(
        train_dates=[pd.Timestamp(d) for d in dates[:n_train]],
        val_dates=[pd.Timestamp(d) for d in dates[n_train : n_train + n_val]],
        test_dates=[pd.Timestamp(d) for d in dates[n_train + n_val :]],
    )


def iter_sessions(df: pd.DataFrame, dates: list[pd.Timestamp]) -> Iterator[tuple[pd.Timestamp, pd.DataFrame]]:
    """Yield (date, sub_df) for each date in `dates`, in order. sub_df has reset index."""
    by_date = {pd.Timestamp(d).date(): g for d, g in df.groupby("session_date", sort=False)}
    for d in dates:
        key = pd.Timestamp(d).date()
        if key in by_date:
            yield d, by_date[key].reset_index(drop=True)


if __name__ == "__main__":
    import yaml

    cfg = yaml.safe_load((WORKSPACE_ROOT / "config.yaml").read_text(encoding="utf-8"))

    df = load_raw(cfg["data"]["path"])
    df = select_window(df, cfg["data"]["window_days"])
    split = split_days(df, cfg["data"]["n_train"], cfg["data"]["n_val"], cfg["data"]["n_test"])

    print(f"Total bars in window: {len(df):,}")
    print(f"Sessions  train={len(split.train_dates)}  val={len(split.val_dates)}  test={len(split.test_dates)}")
    print(f"Train range : {split.train_dates[0].date()}  ->  {split.train_dates[-1].date()}")
    print(f"Val   range : {split.val_dates[0].date()}  ->  {split.val_dates[-1].date()}")
    print(f"Test  range : {split.test_dates[0].date()}  ->  {split.test_dates[-1].date()}")

    lengths = [len(g) for _, g in iter_sessions(df, split.train_dates)]
    print(f"Train session length: min={min(lengths)} median={sorted(lengths)[len(lengths)//2]} max={max(lengths)}")
