"""Baseline trading strategies for final comparison (proposal §Final Baseline Comparison).

All baselines use the same IntradayTradingEnv (same execution model, transaction cost,
and end-of-day flattening) as the DRL agents. Each baseline implements:

    select_action(obs, step, deterministic) -> int

so it is a drop-in for evaluate_policy_per_session() from train.py.

Available baselines:
  FlatBaseline        — always hold no position (action = flat index)
  LongBaseline        — enter long at bar 0, hold until EOD
  ShortBaseline       — enter short at bar 0, hold until EOD
  RandomBaseline      — uniformly random action each bar
  MACrossoverBaseline — long when fast MA > slow MA, short otherwise, flat on cross

MA crossover uses simple moving averages computed on the close price of the current
day's DataFrame. The fast and slow window lengths are configurable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _flat_action_index(cfg: dict) -> int:
    """Return the action index whose position value is 0 (flat)."""
    positions = list(cfg["env"]["action_space"])
    for i, p in enumerate(positions):
        if float(p) == 0.0:
            return i
    raise ValueError("action_space has no flat (0) position — cannot build flat baseline.")


def _long_action_index(cfg: dict) -> int:
    positions = [float(p) for p in cfg["env"]["action_space"]]
    return int(np.argmax(positions))


def _short_action_index(cfg: dict) -> int:
    positions = [float(p) for p in cfg["env"]["action_space"]]
    return int(np.argmin(positions))


# ---------------------------------------------------------------------------
# baseline classes
# ---------------------------------------------------------------------------

class FlatBaseline:
    """Always hold no position. Zero-trade baseline."""

    name = "flat"

    def __init__(self, cfg: dict):
        self._flat = _flat_action_index(cfg)

    def select_action(self, obs: np.ndarray, step: int = 0, deterministic: bool = True) -> int:
        return self._flat

    def reset_episode(self) -> None:
        pass


class LongBaseline:
    """Enter long on the first bar of each session, hold until EOD."""

    name = "long"

    def __init__(self, cfg: dict):
        self._long = _long_action_index(cfg)

    def select_action(self, obs: np.ndarray, step: int = 0, deterministic: bool = True) -> int:
        return self._long

    def reset_episode(self) -> None:
        pass


class ShortBaseline:
    """Enter short on the first bar of each session, hold until EOD."""

    name = "short"

    def __init__(self, cfg: dict):
        self._short = _short_action_index(cfg)

    def select_action(self, obs: np.ndarray, step: int = 0, deterministic: bool = True) -> int:
        return self._short

    def reset_episode(self) -> None:
        pass


class RandomBaseline:
    """Uniformly random action each bar, same action space as the DRL agent."""

    name = "random"

    def __init__(self, cfg: dict, seed: int = 0):
        self._n = len(cfg["env"]["action_space"])
        self._rng = np.random.default_rng(seed)

    def select_action(self, obs: np.ndarray, step: int = 0, deterministic: bool = True) -> int:
        return int(self._rng.integers(0, self._n))

    def reset_episode(self) -> None:
        pass


class MACrossoverBaseline:
    """Moving average crossover: long when fast MA > slow MA, short when fast < slow.

    Signal is computed once per session on the full day's close prices and then
    replayed bar by bar — no future leakage because the signal at bar t uses only
    close[0..t] (the MA is computed with min_periods so early bars are handled).

    Parameters
    ----------
    fast : int  — fast MA window (default 20)
    slow : int  — slow MA window (default 60)
    """

    name = "ma_crossover"

    def __init__(self, cfg: dict, fast: int = 20, slow: int = 60):
        self._long = _long_action_index(cfg)
        self._short = _short_action_index(cfg)
        self._flat = _flat_action_index(cfg)
        self.fast = fast
        self.slow = slow
        self._signals: list[int] = []
        self._t = 0

    def prepare(self, day_df: pd.DataFrame) -> None:
        """Pre-compute bar-level signals for one session. Call before stepping the env."""
        close = day_df["close"].to_numpy(dtype=np.float64)
        n = len(close)
        fast_ma = pd.Series(close).rolling(self.fast, min_periods=1).mean().to_numpy()
        slow_ma = pd.Series(close).rolling(self.slow, min_periods=1).mean().to_numpy()

        signals: list[int] = []
        for i in range(n):
            if fast_ma[i] > slow_ma[i]:
                signals.append(self._long)
            elif fast_ma[i] < slow_ma[i]:
                signals.append(self._short)
            else:
                signals.append(self._flat)
        self._signals = signals
        self._t = 0

    def select_action(self, obs: np.ndarray, step: int = 0, deterministic: bool = True) -> int:
        if self._t < len(self._signals):
            a = self._signals[self._t]
        else:
            a = self._flat
        self._t += 1
        return a

    def reset_episode(self) -> None:
        self._signals = []
        self._t = 0
