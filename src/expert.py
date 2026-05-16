"""Hindsight expert action labeler for BC warm-start (Phase 1d, PROPOSAL Sec. 8).

For each bar t in a session, compute the optimal direction `a in action_space`
using a fixed h-bar lookahead:

    direction[t] = sign(close[t+h] - close[t])  (within +/- noise_threshold)

Then map sign -> nearest action in cfg["env"]["action_space"]. For the default
[-1, 0, +1]: positive future return -> long (action_idx 2), negative -> short
(action_idx 0), |return| < threshold -> flat (action_idx 1).

This is a TRAINING-ONLY signal -- the env never sees expert actions. Trainers
read the expert sequence per session and supply it to the BC loss alongside
each rollout. Live deployment uses the policy directly (no expert needed).

Design choices:
- `h` is matched to the GAE effective horizon. With gamma=0.30, gamma^h ~ 0.05
  at h=10; we default to h=5 (more aggressive). Configurable.
- `noise_threshold` filters out micro-moves that aren't worth the spread cost.
  Defaults to 0.05% of price (about 1-2 dollar moves on XAUUSD).
- For the last `h` bars of each session, the expert is forced to `flat`
  (action_idx of position 0) since lookahead is unavailable. This matches the
  EOD-flat constraint in the env.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_expert_actions(
    day_df: pd.DataFrame,
    action_space: list[float],
    h: int = 5,
    noise_threshold: float = 0.0005,
) -> np.ndarray:
    """Return per-bar expert action indices of shape (n_bars,).

    Parameters
    ----------
    day_df : pd.DataFrame
        One session's DataFrame from features.build_features. Must contain `close`.
    action_space : list[float]
        cfg["env"]["action_space"]. Each entry is a target position in {-1..+1}.
    h : int
        Lookahead horizon in bars. Default 5 (matches gamma=0.3 effective horizon).
    noise_threshold : float
        Fractional move below which we label "flat". Default 0.0005 (0.05%).

    Returns
    -------
    np.ndarray of int64, shape (n_bars,), values in [0, len(action_space)-1].
    """
    close = day_df["close"].to_numpy(dtype=np.float64)
    n = len(close)
    if n < 3:
        raise ValueError(f"Day too short for expert: {n} bars")

    # Future close at t+h (last h bars: no lookahead -> 0 delta -> flat).
    future = np.zeros_like(close)
    future[: n - h] = close[h:]
    future[n - h :] = close[n - h :]  # keep == current so delta = 0

    delta_frac = (future - close) / np.maximum(close, 1e-9)

    # Map each action_space entry to an integer index. Find nearest action
    # to each direction signal.
    actions = np.asarray(action_space, dtype=np.float64)
    flat_idx = int(np.argmin(np.abs(actions)))  # action closest to 0

    # Direction: +1 / 0 / -1 based on noise_threshold
    direction = np.zeros(n, dtype=np.float64)
    direction[delta_frac > noise_threshold] = +1.0
    direction[delta_frac < -noise_threshold] = -1.0

    # For each bar, pick the action_space entry closest to the direction.
    # Build a lookup: argmin(|action - direction|) over actions.
    # Use broadcasting: shape (n, 1) - shape (1, A) = (n, A)
    diff = np.abs(direction[:, None] - actions[None, :])
    expert_idx = np.argmin(diff, axis=1)

    # Override the last h bars to be flat (EOD-flat constraint).
    expert_idx[n - h :] = flat_idx

    return expert_idx.astype(np.int64)


def expert_summary(expert_idx: np.ndarray, action_space: list[float]) -> dict:
    """Return distribution of expert actions for diagnostics."""
    counts = np.bincount(expert_idx, minlength=len(action_space))
    total = max(int(counts.sum()), 1)
    return {
        f"act{i}_pos{action_space[i]:+g}": int(c) for i, c in enumerate(counts)
    } | {"total": total, "frac_flat": float(counts[int(np.argmin(np.abs(action_space)))] / total)}


if __name__ == "__main__":
    # Smoke test: compute expert on the first val session and print stats.
    import sys
    from pathlib import Path

    import yaml

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from data import iter_sessions, load_raw, select_window, split_days
    from features import build_features

    cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text(encoding="utf-8"))
    df = load_raw(cfg["data"]["path"])
    df = select_window(df, cfg["data"]["window_days"])
    feat = build_features(df, cfg["features"])
    split = split_days(feat, cfg["data"]["n_train"], cfg["data"]["n_val"], cfg["data"]["n_test"])

    action_space = cfg["env"]["action_space"]
    print(f"action_space: {action_space}")
    for i, (day, day_df) in enumerate(iter_sessions(feat, split.val_dates[:3])):
        for h in [3, 5, 10]:
            expert = compute_expert_actions(day_df, action_space, h=h, noise_threshold=0.0005)
            stats = expert_summary(expert, action_space)
            print(f"  day {day.date()} bars={len(day_df)} h={h:>2}: {stats}")
        print()
