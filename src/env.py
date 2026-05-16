"""Gymnasium env for intraday XAUUSD per PROPOSAL.md Sec. 3 (portfolio MDP).

- 1 episode  = 1 trading day (variable length).
- 1 step     = 1 M1 bar.
- Action     = index into cfg["env"]["action_space"] list (target-position semantics).
               Default list [-1, 0, +1] gives 3-action short/flat/long.
               Experiment 2 can use [-1, -0.5, 0, +0.5, +1] (5 actions) etc.
- Capital    = $10,000 reset each episode; episodes are independent.
- Lot        = fixed 0.01; contract = 100 oz/lot -> $1 move on gold = $1 per 0.01 lot.
- Cost       = spread only, in dollars, charged on every position change.
- Reward     = R1 (log-return of equity) | R2 (dollar P&L) | R4 (P&L - DD penalty),
               selected via cfg["env"]["reward"]["mode"]. Optional running-std
               normalization (cfg["env"]["reward"]["normalize"]) for training only.
- Ruin       = equity <= 0 at any bar terminates the episode with fixed reward = -1.
- EOD        = force-flatten at the last bar's close, spread cost applied.

State = (10 + 5 + 1)-dim flat vector (10 market + 5 positional + 1 equity_ratio).
No sequence, no LSTM. Observation shape does NOT change with action space size.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from features import MARKET_COLUMNS


POSITIONAL_COLUMNS = ["tl", "pos", "pr", "dr", "ht"]
PORTFOLIO_COLUMNS = ["equity_ratio"]
STATE_COLUMNS = MARKET_COLUMNS + POSITIONAL_COLUMNS + PORTFOLIO_COLUMNS  # 16-dim


@dataclass
class StepInfo:
    position: float         # position sign held over this step (set BEFORE step)
    next_position: float    # position sign held over the next step (set AFTER step)
    p_exec: float           # execution price applied this step
    cost: float             # transaction cost in dollars paid this step
    pnl_dollar: float       # raw dollar Δequity this step (after cost)
    pnl_log: float          # log(equity_{t+1}/equity_t) this step (0 on ruin clip)
    reward: float           # actual scalar fed to optimizer (after mode & norm)
    equity: float           # equity AFTER this step (dollars)
    forced: bool            # True if this step is the forced EOD close
    ruin: bool              # True if episode ended via equity<=0


class RunningStd:
    """Welford running mean/std for reward normalization (training-only)."""

    def __init__(self, eps: float = 1e-8):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.eps = eps

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def std(self) -> float:
        if self.n < 2:
            return 1.0
        return float(np.sqrt(self.m2 / (self.n - 1)) + self.eps)


class IntradayTradingEnv(gym.Env):
    """One-session env. Pass a day's DataFrame (built by features.build_features).

    Portfolio simulation (PROPOSAL.md Sec. 3): fixed starting capital, fixed lot,
    spread-only cost, ruin termination, R1/R2/R4 reward modes with optional
    running-std normalization shared across episodes (training only).

    The running-std normalizer (cfg["env"]["reward"]["normalize"]) should be a
    SHARED `RunningStd` instance across all training episodes for one run, and
    None during evaluation. Pass it via `reward_normalizer` to __init__.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        day_df: pd.DataFrame,
        cfg: dict,
        reward_normalizer: RunningStd | None = None,
    ):
        super().__init__()
        self.cfg = cfg

        env_cfg = cfg["env"]
        # ---- portfolio / contract config -----------------------------------
        self.capital0 = float(env_cfg.get("capital", 10_000.0))
        self.lot = float(env_cfg.get("lot", 0.01))
        # contract_size: oz per standard lot (XAUUSD = 100)
        contract_size = float(env_cfg.get("contract_size", 100.0))
        # $/price unit per 0.01 lot: $1 price move * 100 oz * 0.01 lot = $1
        # General form: dollar_per_price_unit = contract_size * lot
        self.dollar_per_price = contract_size * self.lot

        # ---- spread cost config --------------------------------------------
        # MT5 XAUUSD: 1 point = 0.01 price unit. spread_pts column -> dollars:
        #   $cost = spread_pts * point_size * contract_size * lot
        # With defaults: spread_pts * 0.01 * 100 * 0.01 = spread_pts * 0.01.
        # So 50 points spread -> $0.50 per side per 0.01 lot.
        self.point_size = float(env_cfg.get("spread_point_size", 0.01))
        self._cost_per_point = self.point_size * contract_size * self.lot  # $ per spread point per position change

        # ---- action space --------------------------------------------------
        positions = list(env_cfg["action_space"])
        self._action_to_pos: list[float] = [float(p) for p in positions]
        n_actions = len(self._action_to_pos)

        # ---- reward config -------------------------------------------------
        reward_cfg = env_cfg.get("reward", {})
        self.reward_mode = str(reward_cfg.get("mode", "r1")).lower()
        if self.reward_mode not in {"r1", "r2", "r4"}:
            raise ValueError(f"reward.mode must be r1/r2/r4, got {self.reward_mode!r}")
        # R4 params
        self.r4_beta = float(reward_cfg.get("beta", 1.0))
        self.r4_dd_thresh = float(reward_cfg.get("dd_thresh", 0.02))
        # ruin clip value (per spec: -1)
        self.ruin_reward = float(reward_cfg.get("ruin_reward", -1.0))
        # normalizer: external running-std (shared across episodes) when training
        self.reward_normalizer = reward_normalizer

        # ---- raw arrays for speed ------------------------------------------
        self.open = day_df["open"].to_numpy(dtype=np.float64)
        self.high = day_df["high"].to_numpy(dtype=np.float64)
        self.low = day_df["low"].to_numpy(dtype=np.float64)
        self.close = day_df["close"].to_numpy(dtype=np.float64)
        self.spread_pts = day_df["spread"].to_numpy(dtype=np.float64)
        self.market_feat = day_df[MARKET_COLUMNS].to_numpy(dtype=np.float32)
        self.n_bars = len(day_df)
        if self.n_bars < 3:
            raise ValueError(f"Day too short: {self.n_bars} bars")

        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(STATE_COLUMNS),), dtype=np.float32,
        )

        # episodic state, set in reset()
        self.t = 0
        self.pos = 0.0         # position sign held entering step t (-1..+1, float for Exp 2)
        self.entry_idx = -1    # bar index where current non-flat position was opened
        self.entry_price = 0.0 # exec price at entry
        self.entry_cost = 0.0  # cost in dollars paid at entry
        self.equity = self.capital0
        self.peak_equity = self.capital0
        self.daily_return = 0.0  # cumulative per-bar log return so far (state feature DR)

    # ------------------------------------------------------------------ helpers

    def _spread_cost_dollars(self, t_exec: int, delta_size: float) -> float:
        """Dollar spread cost for a position change of `delta_size` (in position
        units, e.g. |Δa| ∈ [0,2]) at bar index `t_exec`."""
        if delta_size <= 0:
            return 0.0
        return float(self.spread_pts[t_exec] * self._cost_per_point * delta_size)

    def _positional_features(self) -> np.ndarray:
        tl = float(self.n_bars - 1 - self.t)
        pos = float(self.pos)
        if self.pos != 0 and self.entry_idx >= 0 and self.entry_price > 0:
            # Unrealized return since entry, net of entry cost, normalized by entry price.
            # entry_cost in dollars; divide by (entry_price * dollar_per_price) to express
            # as a return fraction comparable to (close - entry_price) / entry_price.
            pr_price = self.pos * (self.close[self.t] - self.entry_price)
            pr_cost_price_units = self.entry_cost / self.dollar_per_price if self.dollar_per_price > 0 else 0.0
            pr = float((pr_price - pr_cost_price_units) / self.entry_price)
            ht = float(self.t - self.entry_idx)
        else:
            pr = 0.0
            ht = 0.0
        return np.array([tl, pos, pr, self.daily_return, ht], dtype=np.float32)

    def _portfolio_features(self) -> np.ndarray:
        equity_ratio = float(self.equity / self.capital0) if self.capital0 > 0 else 0.0
        return np.array([equity_ratio], dtype=np.float32)

    def _obs(self) -> np.ndarray:
        m = self.market_feat[self.t]
        p = self._positional_features()
        e = self._portfolio_features()
        return np.concatenate([m, p, e], axis=0).astype(np.float32)

    # ------------------------------------------------------------------ API

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.t = 0
        self.pos = 0.0
        self.entry_idx = -1
        self.entry_price = 0.0
        self.entry_cost = 0.0
        self.equity = self.capital0
        self.peak_equity = self.capital0
        self.daily_return = 0.0
        return self._obs(), {}

    def step(self, action: int):
        # Episode timeline (PROPOSAL.md Sec. 3.3):
        #   At bar t we observed market_feat[t] and held position `pos` (set last step).
        #   Choose action a_t -> next_pos. Execute at open[t+1] if next_pos != pos,
        #   otherwise mark-to-close at close[t+1]. On the last bar we cannot fill
        #   next-bar, so force-flatten at close[t] (spread cost charged if pos != 0).
        is_last = self.t >= self.n_bars - 1

        if is_last:
            next_pos = 0.0
            forced = True
            p_exec = self.close[self.t]
            p_next_close = self.close[self.t]
            t_exec = self.t
        else:
            next_pos = self._action_to_pos[int(action)]
            forced = False
            if next_pos != self.pos:
                p_exec = self.open[self.t + 1]
                t_exec = self.t + 1
            else:
                p_exec = self.close[self.t]
                t_exec = self.t  # no fill; no cost. t_exec unused
            p_next_close = self.close[self.t + 1]

        # ---- transaction cost (dollars) ------------------------------------
        delta_size = abs(next_pos - self.pos)
        if delta_size > 0:
            cost_dollars = self._spread_cost_dollars(t_exec, delta_size)
        else:
            cost_dollars = 0.0

        # ---- dollar P&L over the next bar ----------------------------------
        # Position held over the next bar is `next_pos`. P&L from price move:
        #   pnl_price = next_pos * (p_next_close - p_exec)  [in price units]
        #   pnl_dollars = pnl_price * dollar_per_price - cost_dollars
        pnl_price = next_pos * (p_next_close - p_exec)
        pnl_dollars = pnl_price * self.dollar_per_price - cost_dollars

        equity_prev = self.equity
        equity_new = equity_prev + pnl_dollars

        # ---- ruin check ----------------------------------------------------
        ruin = equity_new <= 0.0
        if ruin:
            # Per spec: terminate with fixed clipped reward = -1. Equity floors at 0.
            equity_new = 0.0
            pnl_log = 0.0
            raw_reward_unscaled = self.ruin_reward
            scaled_reward = self.ruin_reward  # ruin clip is NOT normalized
        else:
            # log return of equity (used for R1 and as state-feature DR accumulator)
            if equity_prev > 0 and equity_new > 0:
                pnl_log = float(np.log(equity_new / equity_prev))
            else:
                pnl_log = 0.0

            # Update peak and drawdown for R4
            peak_new = max(self.peak_equity, equity_new)
            dd_t = 0.0 if peak_new <= 0 else (peak_new - equity_new) / peak_new

            if self.reward_mode == "r1":
                raw_reward_unscaled = pnl_log
            elif self.reward_mode == "r2":
                raw_reward_unscaled = pnl_dollars
            else:  # r4
                penalty = self.r4_beta * max(0.0, dd_t - self.r4_dd_thresh)
                raw_reward_unscaled = pnl_dollars - penalty

            # Reward normalization (training only). Normalizer state is updated
            # on every non-ruin step; ruin steps emit the fixed clip un-normalized.
            if self.reward_normalizer is not None:
                self.reward_normalizer.update(raw_reward_unscaled)
                scaled_reward = float(raw_reward_unscaled / self.reward_normalizer.std)
            else:
                scaled_reward = float(raw_reward_unscaled)

            # Commit peak update only when not ruined
            self.peak_equity = peak_new

        # ---- entry tracking update -----------------------------------------
        if next_pos != self.pos:
            if next_pos != 0:
                self.entry_idx = t_exec
                self.entry_price = p_exec
                self.entry_cost = cost_dollars
            else:
                self.entry_idx = -1
                self.entry_price = 0.0
                self.entry_cost = 0.0

        # ---- bookkeeping ---------------------------------------------------
        self.equity = equity_new
        self.daily_return += pnl_log

        info = StepInfo(
            position=self.pos,
            next_position=next_pos,
            p_exec=p_exec,
            cost=cost_dollars,
            pnl_dollar=pnl_dollars,
            pnl_log=pnl_log,
            reward=scaled_reward,
            equity=equity_new,
            forced=forced,
            ruin=ruin,
        )

        self.pos = next_pos
        self.t += 1

        terminated = is_last or ruin
        truncated = False
        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._obs()
        return obs, scaled_reward, terminated, truncated, info.__dict__


if __name__ == "__main__":
    from pathlib import Path
    import yaml

    from data import iter_sessions, load_raw, select_window, split_days
    from features import build_features

    cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text(encoding="utf-8"))
    df = load_raw(cfg["data"]["path"])
    df = select_window(df, cfg["data"]["window_days"])
    feat = build_features(df, cfg["features"])
    split = split_days(feat, cfg["data"]["n_train"], cfg["data"]["n_val"], cfg["data"]["n_test"])

    day_date, day_df = next(iter_sessions(feat, split.val_dates[:1]))

    # shared normalizer (would be one-per-run in train.py)
    norm = RunningStd()
    env = IntradayTradingEnv(day_df, cfg, reward_normalizer=norm)
    obs, _ = env.reset()
    print(f"Day {day_date.date()} bars={env.n_bars} obs_shape={obs.shape}  state_dim={len(STATE_COLUMNS)}")
    print(f"  capital={env.capital0}  lot={env.lot}  $/price={env.dollar_per_price}  $/spread_pt={env._cost_per_point}")
    print(f"  reward_mode={env.reward_mode}  beta={env.r4_beta}  dd_thresh={env.r4_dd_thresh}")
    print(f"  obs[:3]={obs[:3]}  equity_ratio={obs[-1]:.6f}")

    rng = np.random.default_rng(0)
    total_reward = 0.0
    trades = 0
    last_pos = 0.0
    n_steps = 0
    while True:
        a = int(rng.integers(0, env.action_space.n))
        obs, r, term, trunc, info = env.step(a)
        total_reward += r
        n_steps += 1
        if info["next_position"] != last_pos:
            trades += 1
            last_pos = info["next_position"]
        if term or trunc:
            print(
                f"  final: t={env.t} forced={info['forced']} ruin={info['ruin']} "
                f"pos_after={info['next_position']} equity=${info['equity']:.2f}"
            )
            break
    print(f"Random rollout: steps={n_steps} total_reward={total_reward:.4f} trades={trades}")
    print(f"Reward normalizer: n={norm.n}  std={norm.std:.6f}")
