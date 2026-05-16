"""Double DQN training loop (portfolio MDP, PROPOSAL.md Sec. 3-4).

Pipeline:
  - Build per-day env stream by cycling through train sessions (one episode = one
    trading day). The portfolio resets each episode (C0 = $10k); ruin terminates.
  - One shared `RunningStd` reward normalizer is wired through DayCycleEnv into
    every train episode (training only). Eval/test envs get None -> no normalization.
  - Run DDQN (see `src/ddqn.py`). The optimizer sees the (possibly normalized)
    scalar; metrics always use `info["pnl_log"]` so Sharpe/Sortino/MDD are
    reward-mode-invariant.
  - Every `eval_every_sessions` completed train episodes, run a deterministic
    rollout over all val sessions and log pooled DeepScalper-style metrics +
    portfolio metrics (final_equity, max_dd_dollar, ruin_rate).
  - Save `best.pt` whenever val `<best_metric>` improves; early stop after
    `early_stop_patience` evals without improvement.
  - Final: load best, evaluate on test, write metrics.csv / trades.csv / summary.json.
"""
from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import yaml

from data import iter_sessions, load_raw, select_window, split_days
from ddqn import DDQNAgent
from env import IntradayTradingEnv, RunningStd, STATE_COLUMNS
from features import build_features

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------- env wrapper ----------------------------


class DayCycleEnv(gym.Env):
    """Cycles train sessions as an infinite single-env stream.

    The agent sees one continuous env; after each session terminates we transparently
    advance to the next session on reset(). Sessions are shuffled at the start of
    each epoch (one epoch = one full pass through `dates`).

    A SINGLE `RunningStd` reward normalizer is passed to every inner env (when
    `reward_normalizer` is non-None) so the running stats persist across episodes
    within a run -- this is what makes R1/R2/R4 comparable at one learning rate.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        feat: pd.DataFrame,
        dates: list[pd.Timestamp],
        cfg: dict,
        shuffle: bool = True,
        seed: int = 0,
        reward_normalizer: RunningStd | None = None,
    ):
        super().__init__()
        self.feat = feat
        self.dates_orig = list(dates)
        self.cfg = cfg
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        self.reward_normalizer = reward_normalizer
        self._build_order()
        self._cursor = 0
        first_day_df = self._first_valid_day()
        self._inner: IntradayTradingEnv = IntradayTradingEnv(
            first_day_df, self.cfg, reward_normalizer=self.reward_normalizer,
        )
        self.action_space = self._inner.action_space
        self.observation_space = self._inner.observation_space
        self.current_date: pd.Timestamp | None = None

    def _build_order(self) -> None:
        self.order = list(self.dates_orig)
        if self.shuffle:
            self.rng.shuffle(self.order)
        self.epoch_count = getattr(self, "epoch_count", 0) + 1 if hasattr(self, "epoch_count") else 1

    def _first_valid_day(self) -> pd.DataFrame:
        for d in self.order:
            it = iter_sessions(self.feat, [d])
            try:
                _, day_df = next(it)
            except StopIteration:
                continue
            if len(day_df) >= 3:
                return day_df
        raise RuntimeError("No valid training session found (all too short or missing).")

    def _advance_inner(self) -> None:
        while True:
            if self._cursor >= len(self.order):
                self._build_order()
                self._cursor = 0
            d = self.order[self._cursor]
            self._cursor += 1
            it = iter_sessions(self.feat, [d])
            try:
                _, day_df = next(it)
            except StopIteration:
                continue
            if len(day_df) < 3:
                continue
            self._inner = IntradayTradingEnv(
                day_df, self.cfg, reward_normalizer=self.reward_normalizer,
            )
            self.current_date = d
            return

    def reset(self, *, seed=None, options=None):
        self._advance_inner()
        return self._inner.reset(seed=seed, options=options)

    def step(self, action):
        assert self._inner is not None
        return self._inner.step(action)


# ---------------------------- metrics ----------------------------


def pooled_metrics(
    returns: np.ndarray,
    trade_pnls: list[float],
    day_lengths: list[int] | None = None,
    positions: np.ndarray | None = None,
    bars_held_per_trade: list[int] | None = None,
    final_equities: list[float] | None = None,
    max_dd_dollars: list[float] | None = None,
    ruin_flags: list[bool] | None = None,
    capital0: float = 10_000.0,
) -> dict[str, float]:
    """DeepScalper §5.2 pooled-period metrics + portfolio metrics (PROPOSAL Sec. 4).

    `returns`          - concatenated per-bar LOG-RETURN sequence across all sessions.
                         IMPORTANT: this is info["pnl_log"], NOT the optimizer reward.
                         This way Sharpe/Sortino/MDD are invariant to reward mode.
    `trade_pnls`       - flat list of completed-trade log-PnLs.
    `day_lengths`      - bar-counts per day; enables daily-annualized Sharpe/Sortino.
                         None -> per-bar fallback with no annualization.
    `positions`        - per-bar position array (same length as returns); used for turnover.
    `bars_held_per_trade` - bars_held for each completed trade; used for avg_holding_time.
    `final_equities`   - per-episode end equity (dollars). None -> 0.
    `max_dd_dollars`   - per-episode worst peak-to-trough equity drop (dollars).
    `ruin_flags`       - per-episode bool: equity<=0 hit.
    `capital0`         - starting capital, used for max_dd_pct.
    """
    if len(returns) == 0:
        return {
            "total_return": 0.0, "mdd": 0.0, "trades": 0, "winrate": 0.0,
            "sharpe": 0.0, "sortino": 0.0,
            "avg_trade_pnl": 0.0, "turnover": 0.0, "avg_holding_time": 0.0,
            "final_equity": 0.0, "max_dd_dollar": 0.0, "max_dd_pct": 0.0, "ruin_rate": 0.0,
        }

    cum = np.cumsum(returns)
    equity_curve = np.exp(cum)
    peak = np.maximum.accumulate(equity_curve)
    mdd = float(((equity_curve - peak) / peak).min())

    total_return = float(math.exp(cum[-1]) - 1.0)

    n_trades = len(trade_pnls)
    if n_trades > 0:
        trade_arr = np.array(trade_pnls)
        winrate = float((trade_arr > 0).sum() / n_trades)
        avg_trade_pnl = float(trade_arr.mean())
    else:
        winrate = 0.0
        avg_trade_pnl = 0.0

    if positions is not None and len(positions) > 1:
        turnover = float(np.abs(np.diff(positions.astype(float))).sum())
    else:
        turnover = 0.0

    if bars_held_per_trade is not None and len(bars_held_per_trade) > 0:
        avg_holding_time = float(np.mean(bars_held_per_trade))
    else:
        avg_holding_time = 0.0

    ANNUALIZE = math.sqrt(252)
    if day_lengths is not None and len(day_lengths) > 0:
        idx = 0
        daily_returns: list[float] = []
        for n in day_lengths:
            daily_returns.append(float(returns[idx: idx + n].sum()))
            idx += n
        dr = np.array(daily_returns)
        mean_d = float(dr.mean())
        std_d = float(dr.std())
        sharpe = mean_d / std_d * ANNUALIZE if std_d > 1e-12 else 0.0
        downside_d = dr[dr < 0]
        dd_std = float(downside_d.std()) if len(downside_d) > 0 else 0.0
        sortino = mean_d / dd_std * ANNUALIZE if dd_std > 1e-12 else 0.0
    else:
        mean_r = float(returns.mean())
        std_r = float(returns.std())
        sharpe = mean_r / std_r if std_r > 1e-12 else 0.0
        downside = returns[returns < 0]
        dd_std = float(downside.std()) if len(downside) > 0 else 0.0
        sortino = mean_r / dd_std if dd_std > 1e-12 else 0.0

    # ---- portfolio metrics (PROPOSAL Sec. 4) ----
    if final_equities is not None and len(final_equities) > 0:
        final_equity = float(np.mean(final_equities))
    else:
        final_equity = 0.0
    if max_dd_dollars is not None and len(max_dd_dollars) > 0:
        max_dd_dollar = float(np.max(max_dd_dollars))  # worst per-episode dollar DD
        max_dd_pct = float(max_dd_dollar / capital0) if capital0 > 0 else 0.0
    else:
        max_dd_dollar = 0.0
        max_dd_pct = 0.0
    if ruin_flags is not None and len(ruin_flags) > 0:
        ruin_rate = float(np.mean(ruin_flags))
    else:
        ruin_rate = 0.0

    return {
        "total_return": total_return,
        "mdd": mdd,
        "trades": int(n_trades),
        "winrate": winrate,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "avg_trade_pnl": avg_trade_pnl,
        "turnover": turnover,
        "avg_holding_time": avg_holding_time,
        "final_equity": final_equity,
        "max_dd_dollar": max_dd_dollar,
        "max_dd_pct": max_dd_pct,
        "ruin_rate": ruin_rate,
    }


def trade_pnls_from_session(returns: list[float], positions: list[float]) -> list[float]:
    """Sum log-returns within each contiguous non-zero position block."""
    out: list[float] = []
    cur_side = 0
    cur_pnl = 0.0
    for r, p in zip(returns, positions):
        if p != cur_side:
            if cur_side != 0:
                out.append(cur_pnl)
            cur_side = int(p)
            cur_pnl = 0.0
        cur_pnl += float(r)
    if cur_side != 0:
        out.append(cur_pnl)
    return out


def extract_trades(times: list, prices_close: list[float], positions: list[float], returns: list[float]) -> list[dict]:
    trades: list[dict] = []
    n = len(positions)
    i = 0
    while i < n:
        if positions[i] == 0:
            i += 1
            continue
        side = positions[i]
        start = i
        pnl = 0.0
        while i < n and positions[i] == side:
            pnl += returns[i]
            i += 1
        end = i - 1
        trades.append({
            "entry_time": times[start],
            "exit_time": times[end],
            "side": "long" if side > 0 else "short",
            "entry_price": prices_close[start],
            "exit_price": prices_close[end],
            "bars_held": end - start + 1,
            "pnl_log": pnl,
        })
    return trades


def episode_max_dd_dollar(equity_series: list[float], capital0: float) -> float:
    """Worst peak-to-trough drop in dollars across one episode.
    Equity series is the equity AFTER each bar; we prepend capital0 as the t=0 mark."""
    if not equity_series:
        return 0.0
    eq = np.concatenate([[capital0], np.asarray(equity_series, dtype=np.float64)])
    peak = np.maximum.accumulate(eq)
    dd = peak - eq  # positive dollars
    return float(dd.max())


# ---------------------------- evaluation ----------------------------


def evaluate_policy_per_session(
    agent,
    feat: pd.DataFrame,
    dates: list[pd.Timestamp],
    cfg: dict,
    trade_log: list[dict] | None = None,
    phase: str = "val",
) -> dict[str, float]:
    """Deterministic greedy rollout over `dates`, pooled DeepScalper-style.

    Eval envs are constructed with `reward_normalizer=None` so evaluation is never
    affected by training-time reward normalization (PROPOSAL Sec. 6.4).

    Per-bar `info["pnl_log"]` is used for the return sequence (NOT the reward),
    which makes the metric set invariant to reward mode (r1/r2/r4).
    """
    capital0 = float(cfg["env"].get("capital", 10_000.0))

    pooled_returns: list[float] = []
    pooled_positions: list[float] = []
    pooled_trade_pnls: list[float] = []
    pooled_bars_held: list[int] = []
    day_lengths: list[int] = []
    final_equities: list[float] = []
    max_dd_dollars: list[float] = []
    ruin_flags: list[bool] = []

    for day_date, day_df in iter_sessions(feat, dates):
        if len(day_df) < 3:
            continue
        env = IntradayTradingEnv(day_df, cfg, reward_normalizer=None)
        if hasattr(agent, "prepare"):
            agent.prepare(day_df)
        obs, _ = env.reset()
        ep_returns: list[float] = []   # info["pnl_log"] per bar (NOT the reward)
        ep_positions: list[float] = []
        ep_equity: list[float] = []
        ep_ruin = False
        while True:
            action = agent.select_action(obs, step=0, deterministic=True)
            obs, _r, term, trunc, info = env.step(action)
            ep_returns.append(float(info["pnl_log"]))
            ep_positions.append(float(info["next_position"]))
            ep_equity.append(float(info["equity"]))
            if info.get("ruin", False):
                ep_ruin = True
            if term or trunc:
                break

        times = day_df["time"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        closes = day_df["close"].tolist()
        # Trim times/closes to actual episode length (ruin may end early).
        ep_n = len(ep_returns)
        day_trades = extract_trades(times[:ep_n], closes[:ep_n], ep_positions, ep_returns)

        pooled_returns.extend(ep_returns)
        pooled_positions.extend(ep_positions)
        pooled_trade_pnls.extend(trade_pnls_from_session(ep_returns, ep_positions))
        pooled_bars_held.extend(tr["bars_held"] for tr in day_trades)
        day_lengths.append(ep_n)
        final_equities.append(ep_equity[-1] if ep_equity else capital0)
        max_dd_dollars.append(episode_max_dd_dollar(ep_equity, capital0))
        ruin_flags.append(ep_ruin)

        if trade_log is not None:
            for tr in day_trades:
                tr["phase"] = phase
                tr["day"] = str(day_date.date())
                trade_log.append(tr)

    return pooled_metrics(
        np.array(pooled_returns),
        pooled_trade_pnls,
        day_lengths=day_lengths,
        positions=np.array(pooled_positions),
        bars_held_per_trade=pooled_bars_held,
        final_equities=final_equities,
        max_dd_dollars=max_dd_dollars,
        ruin_flags=ruin_flags,
        capital0=capital0,
    )


# ---------------------------- main entry ----------------------------


# metrics.csv schema (kept stable so downstream tools can rely on it)
METRICS_HEADER = (
    "episode,global_step,epoch,phase,total_return,mdd,trades,winrate,sharpe,sortino,"
    "avg_trade_pnl,turnover,avg_holding_time,final_equity,max_dd_dollar,max_dd_pct,ruin_rate\n"
)
METRICS_COLS = [
    "total_return", "mdd", "trades", "winrate", "sharpe", "sortino",
    "avg_trade_pnl", "turnover", "avg_holding_time",
    "final_equity", "max_dd_dollar", "max_dd_pct", "ruin_rate",
]


def _row_from_metrics(episode, gs, epoch, phase, m: dict) -> list:
    return [episode, gs, epoch, phase] + [m[k] for k in METRICS_COLS]


def train_ddqn(cfg: dict) -> dict:
    seed = int(cfg["train"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device_str = "cuda" if (cfg["train"]["device"] == "cuda" and torch.cuda.is_available()) else "cpu"
    device = torch.device(device_str)
    print(f"Device: {device_str}")

    df = load_raw(cfg["data"]["path"])
    df = select_window(df, cfg["data"]["window_days"])
    feat = build_features(df, cfg["features"])
    split = split_days(feat, cfg["data"]["n_train"], cfg["data"]["n_val"], cfg["data"]["n_test"])

    hpo = cfg.get("_hpo")
    if hpo is not None:
        train_dates = list(hpo["inner_train_dates"])
        val_dates = list(hpo["inner_val_dates"])
    else:
        train_dates = split.train_dates
        val_dates = split.val_dates
    print(f"Sessions  train/val/test = {len(train_dates)}/{len(val_dates)}/{len(split.test_dates)}"
          + ("  [HPO inner-CV fold]" if hpo is not None else ""))

    run_dir = (WORKSPACE_ROOT / cfg["run"]["output_root"] / cfg["run"]["run_name"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_dump = {k: v for k, v in cfg.items() if k != "_hpo"}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg_dump, sort_keys=False), encoding="utf-8")

    metrics_path = run_dir / "metrics.csv"
    metrics_path.write_text(METRICS_HEADER, encoding="utf-8")
    (run_dir / "trades.csv").write_text(
        "episode,phase,day,entry_time,exit_time,side,entry_price,exit_price,bars_held,pnl_log\n",
        encoding="utf-8",
    )
    tb_dir = (WORKSPACE_ROOT / cfg["run"]["output_root"] / "_tb" / cfg["run"]["run_name"]).resolve()
    tb_dir.mkdir(parents=True, exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter
    tb_writer = SummaryWriter(str(tb_dir))

    # Shared running-std reward normalizer (training only). Eval envs get None.
    reward_cfg = cfg["env"].get("reward", {})
    use_norm = bool(reward_cfg.get("normalize", True))
    reward_normalizer = RunningStd() if use_norm else None

    train_env = DayCycleEnv(
        feat, train_dates, cfg, shuffle=True, seed=seed,
        reward_normalizer=reward_normalizer,
    )
    sessions_per_epoch = len(train_dates)
    obs_dim = int(train_env.observation_space.shape[0])
    n_actions = int(train_env.action_space.n)
    capital0 = float(cfg["env"].get("capital", 10_000.0))

    dqn_cfg = cfg["dqn"]
    total_timesteps = int(hpo["timesteps_override"]) if (hpo is not None and "timesteps_override" in hpo) \
        else int(cfg["train"]["total_timesteps"])
    agent = DDQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_sizes=list(dqn_cfg["hidden_sizes"]),
        lr=float(dqn_cfg["lr"]),
        gamma=float(dqn_cfg["gamma"]),
        buffer_size=int(dqn_cfg["buffer_size"]),
        batch_size=int(dqn_cfg["batch_size"]),
        learning_starts=int(dqn_cfg["learning_starts"]),
        train_freq=int(dqn_cfg["train_freq"]),
        gradient_steps=int(dqn_cfg["gradient_steps"]),
        target_update_interval=int(dqn_cfg["target_update_interval"]),
        exploration_fraction=float(dqn_cfg["exploration_fraction"]),
        exploration_initial_eps=float(dqn_cfg["exploration_initial_eps"]),
        exploration_final_eps=float(dqn_cfg["exploration_final_eps"]),
        max_grad_norm=float(dqn_cfg["max_grad_norm"]),
        total_timesteps=total_timesteps,
        device=device,
        seed=seed,
    )

    best_metric = str(cfg["train"].get("best_metric", "sortino")).lower()
    allowed = {"total_return", "sharpe", "sortino", "final_equity"}
    if best_metric not in allowed:
        raise ValueError(f"train.best_metric must be one of {allowed}, got {best_metric!r}")
    early_stop_patience = int(cfg["train"]["early_stop_patience"])
    eval_every_sessions = max(1, int(cfg["train"]["eval_every_sessions"]))
    log_interval = int(cfg["train"].get("log_interval", 10))

    best_value = -math.inf
    evals_since_improve = 0
    eval_count = 0
    train_episodes_done = 0

    # Per-episode collectors (logs use info["pnl_log"] for metric-invariance to reward mode)
    cur_ep_returns: list[float] = []   # pnl_log per bar
    cur_ep_positions: list[float] = []
    cur_ep_equity: list[float] = []
    cur_ep_ruin = False
    cur_ep_reward_sum = 0.0  # tracks the optimizer's scalar (for TB only)

    t0 = time.time()
    obs, _ = train_env.reset()
    early_stop = False

    for step in range(1, total_timesteps + 1):
        action = agent.select_action(obs, step=step - 1, deterministic=False)
        next_obs, reward, terminated, truncated, info = train_env.step(action)
        done = bool(terminated)

        agent.replay.add(obs, action, reward, next_obs, done)
        cur_ep_returns.append(float(info["pnl_log"]))
        cur_ep_positions.append(float(info["next_position"]))
        cur_ep_equity.append(float(info["equity"]))
        if info.get("ruin", False):
            cur_ep_ruin = True
        cur_ep_reward_sum += float(reward)

        obs = next_obs

        agent.maybe_learn(step)
        agent.maybe_update_target(step)

        if terminated or truncated:
            train_episodes_done += 1
            gs = step
            epoch_idx = (train_episodes_done - 1) // sessions_per_epoch + 1
            returns_arr = np.array(cur_ep_returns, dtype=np.float64)
            positions_arr = np.array(cur_ep_positions)
            trade_pnls = trade_pnls_from_session(cur_ep_returns, cur_ep_positions)
            m = pooled_metrics(
                returns_arr, trade_pnls,
                positions=positions_arr,
                final_equities=[cur_ep_equity[-1]] if cur_ep_equity else [capital0],
                max_dd_dollars=[episode_max_dd_dollar(cur_ep_equity, capital0)],
                ruin_flags=[cur_ep_ruin],
                capital0=capital0,
            )

            for k, v in m.items():
                tb_writer.add_scalar(f"train/{k}", v, gs)
            tb_writer.add_scalar("train/episode_reward", cur_ep_reward_sum, gs)
            tb_writer.add_scalar("train/episode_length", int(len(returns_arr)), gs)
            tb_writer.add_scalar("train/episode_count", train_episodes_done, gs)
            tb_writer.add_scalar("train/epoch", epoch_idx, gs)
            tb_writer.add_scalar("train/exploration_rate", agent.current_eps(step), gs)
            if reward_normalizer is not None:
                tb_writer.add_scalar("train/reward_norm_std", reward_normalizer.std, gs)
            if not math.isnan(agent.last_loss):
                tb_writer.add_scalar("train/loss", agent.last_loss, gs)
            if not math.isnan(agent.last_mean_q):
                tb_writer.add_scalar("train/mean_q", agent.last_mean_q, gs)

            with metrics_path.open("a", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(_row_from_metrics(train_episodes_done, gs, epoch_idx, "train", m))

            if train_episodes_done % log_interval == 0:
                ruin_tag = " RUIN" if cur_ep_ruin else ""
                print(f"[ep {train_episodes_done:04d}] step={gs} eps={agent.current_eps(step):.3f}  "
                      f"loss={agent.last_loss:.4g} meanQ={agent.last_mean_q:+.3f}  "
                      f"eq=${m['final_equity']:.0f} dd=${m['max_dd_dollar']:.0f} "
                      f"trades={m['trades']}{ruin_tag}")

            cur_ep_returns = []
            cur_ep_positions = []
            cur_ep_equity = []
            cur_ep_ruin = False
            cur_ep_reward_sum = 0.0

            if train_episodes_done % eval_every_sessions == 0:
                eval_count += 1
                trades_buf: list[dict] = []
                vm = evaluate_policy_per_session(
                    agent, feat, val_dates, cfg, trade_log=trades_buf, phase="val",
                )
                with metrics_path.open("a", newline="", encoding="utf-8") as fh:
                    csv.writer(fh).writerow(_row_from_metrics(train_episodes_done, gs, epoch_idx, "val", vm))
                for k, v in vm.items():
                    tb_writer.add_scalar(f"val/{k}", v, gs)

                trades_path = run_dir / "trades.csv"
                with trades_path.open("a", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    for tr in trades_buf:
                        w.writerow([train_episodes_done, tr["phase"], tr["day"], tr["entry_time"], tr["exit_time"],
                                    tr["side"], tr["entry_price"], tr["exit_price"], tr["bars_held"], tr["pnl_log"]])

                cur_value = vm[best_metric]
                improved = cur_value > best_value
                if improved:
                    best_value = cur_value
                    evals_since_improve = 0
                    agent.save(run_dir / "best.pt")
                    best_val_trades = int(vm["trades"])
                    (run_dir / "best_info.json").write_text(json.dumps({
                        "episode": train_episodes_done,
                        "epoch": epoch_idx,
                        "eval_count": eval_count,
                        "global_step": gs,
                        "metric": best_metric,
                        f"val_{best_metric}": best_value,
                        "val_trades": best_val_trades,
                        "val_final_equity": float(vm["final_equity"]),
                        "val_ruin_rate": float(vm["ruin_rate"]),
                    }, indent=2), encoding="utf-8")
                    patience_tag = f"NEW BEST {best_metric}={best_value:+.4f}"
                else:
                    evals_since_improve += 1
                    patience_tag = (f"no improve {evals_since_improve}/{early_stop_patience}  "
                                    f"(best {best_metric}={best_value:+.4f})")

                print(f"[eval {eval_count:03d}] ep={train_episodes_done:04d} epoch={epoch_idx} step={gs}  "
                      f"val: eq=${vm['final_equity']:.0f} ret={vm['total_return']:+.4f} "
                      f"sharpe={vm['sharpe']:+.3f} sortino={vm['sortino']:+.3f} "
                      f"mdd=${vm['max_dd_dollar']:.0f} trades={vm['trades']} ruin={vm['ruin_rate']:.2f}  "
                      f"|  {patience_tag}")

                if not improved and evals_since_improve >= early_stop_patience:
                    print(f"Early stop at episode {train_episodes_done} (eval {eval_count}): "
                          f"{evals_since_improve} consecutive evals with no val {best_metric} improvement.")
                    early_stop = True

            obs, _ = train_env.reset()
            if early_stop:
                break

    total_elapsed = time.time() - t0
    print(f"Training done in {total_elapsed:.1f}s, episodes={train_episodes_done}, steps={step}")

    if hpo is not None:
        tb_writer.close()
        bi = (
            json.loads((run_dir / "best_info.json").read_text(encoding="utf-8"))
            if (run_dir / "best_info.json").exists() else {}
        )
        val_trades = int(bi.get("val_trades", 0))
        print(f"[HPO fold] best inner-val {best_metric}={best_value:+.6f} val_trades={val_trades}")
        return {
            "hpo_objective": float(best_value),
            "best_metric": best_metric,
            "val_trades": val_trades,
        }

    # --- final test ---
    best_path = run_dir / "best.pt"
    if best_path.exists():
        agent.load(best_path)
        print(f"Loaded best checkpoint from {best_path}")
    else:
        print("No best checkpoint saved; using final agent for test eval.")

    test_trades: list[dict] = []
    test_m = evaluate_policy_per_session(
        agent, feat, split.test_dates, cfg, trade_log=test_trades, phase="test",
    )
    gs_final = step
    for k, v in test_m.items():
        tb_writer.add_scalar(f"test/{k}", v, gs_final)

    best_info = (
        json.loads((run_dir / "best_info.json").read_text(encoding="utf-8"))
        if (run_dir / "best_info.json").exists()
        else {}
    )
    best_epoch = best_info.get("epoch", "")
    with metrics_path.open("a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(_row_from_metrics("BEST", gs_final, best_epoch, "test", test_m))

    with (run_dir / "trades.csv").open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for tr in test_trades:
            w.writerow(["BEST", tr["phase"], tr["day"], tr["entry_time"], tr["exit_time"],
                        tr["side"], tr["entry_price"], tr["exit_price"], tr["bars_held"], tr["pnl_log"]])

    summary = {
        "run_name": cfg["run"]["run_name"],
        "best_checkpoint": {**best_info, "path": str(best_path.resolve()) if best_path.exists() else None},
        "training": {
            "total_episodes": train_episodes_done,
            "total_epochs": (train_episodes_done - 1) // sessions_per_epoch + 1 if train_episodes_done > 0 else 0,
            "total_timesteps": step,
            "total_evals": eval_count,
            "sessions_per_epoch": sessions_per_epoch,
            "eval_every_sessions": eval_every_sessions,
            "n_train_sessions": len(split.train_dates),
            "n_val_sessions": len(split.val_dates),
            "n_test_sessions": len(split.test_dates),
            "wall_seconds": total_elapsed,
            "reward_mode": str(cfg["env"].get("reward", {}).get("mode", "r1")),
            "reward_normalize": bool(reward_cfg.get("normalize", True)),
        },
        "test_metrics": test_m,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if best_info:
        metric_name = best_info.get("metric", best_metric)
        metric_val = best_info.get(f"val_{metric_name}", float("nan"))
        print(f"\n[BEST] eval={best_info.get('eval_count', '?')} "
              f"episode={best_info.get('episode', '?')} "
              f"epoch={best_info.get('epoch', '?')} "
              f"step={best_info.get('global_step', '?')}  "
              f"val_{metric_name}={metric_val:+.4f}")
    else:
        print(f"\n[BEST] no checkpoint saved (val {best_metric} never improved above -inf)")
    print(f"[TEST] eq=${test_m['final_equity']:.2f}  ret={test_m['total_return']:+.4f}  "
          f"sharpe={test_m['sharpe']:+.3f}  sortino={test_m['sortino']:+.3f}  "
          f"mdd=${test_m['max_dd_dollar']:.0f} ({test_m['max_dd_pct']*100:.2f}%)  "
          f"trades={test_m['trades']}  winrate={test_m['winrate']:.3f}  "
          f"ruin={test_m['ruin_rate']:.2f}")

    tb_writer.add_text("config", "```yaml\n" + yaml.safe_dump(cfg_dump, sort_keys=False) + "\n```", 0)
    hparams = {
        "algo": "ddqn",
        "run_name": str(cfg["run"]["run_name"]),
        "seed": int(cfg["train"]["seed"]),
        "lr": float(dqn_cfg["lr"]),
        "batch_size": int(dqn_cfg["batch_size"]),
        "target_update_interval": int(dqn_cfg["target_update_interval"]),
        "exploration_fraction": float(dqn_cfg["exploration_fraction"]),
        "hidden_sizes": str(list(dqn_cfg["hidden_sizes"])),
        "best_metric": best_metric,
        "reward_mode": str(cfg["env"].get("reward", {}).get("mode", "r1")),
        "total_timesteps": int(cfg["train"]["total_timesteps"]),
    }
    hmetrics = {
        f"hparam/val_{best_metric}": float(best_info.get(f"val_{best_metric}", float("nan"))) if best_info else float("nan"),
        "hparam/test_total_return": float(test_m["total_return"]),
        "hparam/test_sharpe": float(test_m["sharpe"]),
        "hparam/test_sortino": float(test_m["sortino"]),
        "hparam/test_final_equity": float(test_m["final_equity"]),
        "hparam/test_max_dd_pct": float(test_m["max_dd_pct"]),
        "hparam/test_ruin_rate": float(test_m["ruin_rate"]),
    }
    tb_writer.add_hparams(hparams, hmetrics, run_name=".")
    tb_writer.close()
    return test_m


if __name__ == "__main__":
    cfg = yaml.safe_load((WORKSPACE_ROOT / "config.yaml").read_text(encoding="utf-8"))
    train_ddqn(cfg)
