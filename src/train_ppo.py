"""PPO training loop. Episode-aligned: 1 rollout = 1 trading day.

Mirrors `src/train_a2c.py` exactly except for the agent and the PPO-specific
diagnostic scalars (approx_kl, clip_fraction, n_epochs_run). Portfolio MDP
support: shared RunningStd reward normalizer across episodes; metrics computed
from `info["pnl_log"]` for reward-mode-invariance.
"""
from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from a2c import Rollout
from data import iter_sessions, load_raw, select_window, split_days
from env import RunningStd
from expert import compute_expert_actions
from features import build_features
from ppo import PPOAgent
from train import (
    DayCycleEnv,
    METRICS_HEADER,
    _row_from_metrics,
    episode_max_dd_dollar,
    evaluate_policy_per_session,
    pooled_metrics,
    trade_pnls_from_session,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def train_ppo(cfg: dict) -> dict:
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

    ppo_cfg = cfg["ppo"]
    bc_cfg = cfg.get("bc", {})
    bc_coef = float(bc_cfg.get("coef", 0.0))
    bc_anneal_steps = int(bc_cfg.get("anneal_steps", 0))
    bc_h = int(bc_cfg.get("lookahead", 5))
    bc_thresh = float(bc_cfg.get("noise_threshold", 0.0005))
    bc_active = bc_coef > 0.0 and bc_anneal_steps > 0
    # LR schedule (Option 1): high lr during BC phase, anneal to base RL lr.
    rl_lr = float(ppo_cfg["lr"])
    bc_lr_raw = bc_cfg.get("lr_bc")
    bc_lr = float(bc_lr_raw) if (bc_active and bc_lr_raw is not None) else rl_lr

    def _bc_lr_at(gstep: int) -> float:
        if not bc_active or bc_anneal_steps <= 0:
            return rl_lr
        frac = min(1.0, max(0.0, float(gstep) / float(bc_anneal_steps)))
        return bc_lr + (rl_lr - bc_lr) * frac

    if bc_active:
        action_space_list = list(cfg["env"]["action_space"])
        print(f"BC warm-start: coef={bc_coef} anneal_steps={bc_anneal_steps} "
              f"lookahead={bc_h} noise_thresh={bc_thresh} lr_bc={bc_lr} rl_lr={rl_lr}")

    agent = PPOAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_sizes=list(ppo_cfg["hidden_sizes"]),
        lr=float(ppo_cfg["lr"]),
        gamma=float(ppo_cfg["gamma"]),
        gae_lambda=float(ppo_cfg["gae_lambda"]),
        clip_range=float(ppo_cfg["clip_range"]),
        clip_range_vf=ppo_cfg.get("clip_range_vf"),
        value_coef=float(ppo_cfg["value_coef"]),
        entropy_coef=float(ppo_cfg["entropy_coef"]),
        max_grad_norm=float(ppo_cfg["max_grad_norm"]),
        n_epochs=int(ppo_cfg["n_epochs"]),
        minibatch_size=int(ppo_cfg["minibatch_size"]),
        target_kl=ppo_cfg.get("target_kl"),
        normalize_advantage=bool(ppo_cfg["normalize_advantage"]),
        device=device,
        seed=seed,
        bc_coef=bc_coef,
        bc_anneal_steps=bc_anneal_steps,
    )

    best_metric = str(cfg["train"].get("best_metric", "sortino")).lower()
    allowed = {"total_return", "sharpe", "sortino", "final_equity"}
    if best_metric not in allowed:
        raise ValueError(f"train.best_metric must be one of {allowed}, got {best_metric!r}")
    early_stop_patience = int(cfg["train"]["early_stop_patience"])
    eval_every_sessions = max(1, int(cfg["train"]["eval_every_sessions"]))
    log_interval = int(cfg["train"].get("log_interval", 10))
    total_timesteps = int(hpo["timesteps_override"]) if (hpo is not None and "timesteps_override" in hpo) \
        else int(cfg["train"]["total_timesteps"])

    best_value = -math.inf
    evals_since_improve = 0
    eval_count = 0
    train_episodes_done = 0

    t0 = time.time()
    early_stop = False
    step = 0
    obs, _ = train_env.reset()
    rollout = Rollout()
    ep_pnl_log: list[float] = []
    ep_equity: list[float] = []
    ep_ruin = False
    cur_expert: np.ndarray | None = None
    if bc_active:
        try:
            _, cur_day_df = next(iter_sessions(feat, [train_env.current_date]))
            cur_expert = compute_expert_actions(cur_day_df, action_space_list,
                                                h=bc_h, noise_threshold=bc_thresh)
        except StopIteration:
            cur_expert = None
    ep_bar_idx = 0

    while step < total_timesteps:
        action, log_prob, value = agent.act_and_evaluate(obs)
        next_obs, reward, terminated, truncated, info = train_env.step(action)
        done = bool(terminated)

        if cur_expert is not None and ep_bar_idx < len(cur_expert):
            expert_a = int(cur_expert[ep_bar_idx])
        else:
            expert_a = -1

        rollout.add(obs, action, log_prob, value, reward, done,
                    next_position=float(info["next_position"]),
                    expert_action=expert_a)
        ep_pnl_log.append(float(info["pnl_log"]))
        ep_equity.append(float(info["equity"]))
        if info.get("ruin", False):
            ep_ruin = True

        obs = next_obs
        step += 1
        ep_bar_idx += 1

        if terminated or truncated:
            if bc_active:
                cur_lr = _bc_lr_at(step)
                for pg in agent.optim.param_groups:
                    pg["lr"] = cur_lr
            agent.update(rollout, last_value=0.0, global_step=step)
            train_episodes_done += 1
            gs = step
            epoch_idx = (train_episodes_done - 1) // sessions_per_epoch + 1

            returns_arr = np.array(ep_pnl_log, dtype=np.float64)
            positions_arr = np.array(rollout.next_positions)
            trade_pnls = trade_pnls_from_session(ep_pnl_log, rollout.next_positions)
            m = pooled_metrics(
                returns_arr, trade_pnls,
                positions=positions_arr,
                final_equities=[ep_equity[-1]] if ep_equity else [capital0],
                max_dd_dollars=[episode_max_dd_dollar(ep_equity, capital0)],
                ruin_flags=[ep_ruin],
                capital0=capital0,
            )

            for k, v in m.items():
                tb_writer.add_scalar(f"train/{k}", v, gs)
            tb_writer.add_scalar("train/episode_reward", float(np.sum(rollout.rewards)), gs)
            tb_writer.add_scalar("train/episode_length", int(len(returns_arr)), gs)
            tb_writer.add_scalar("train/episode_count", train_episodes_done, gs)
            tb_writer.add_scalar("train/epoch", epoch_idx, gs)
            if reward_normalizer is not None:
                tb_writer.add_scalar("train/reward_norm_std", reward_normalizer.std, gs)
            if not math.isnan(agent.last_policy_loss):
                tb_writer.add_scalar("train/policy_loss", agent.last_policy_loss, gs)
                tb_writer.add_scalar("train/value_loss", agent.last_value_loss, gs)
                tb_writer.add_scalar("train/entropy", agent.last_entropy, gs)
                tb_writer.add_scalar("train/total_loss", agent.last_total_loss, gs)
                tb_writer.add_scalar("train/approx_kl", agent.last_approx_kl, gs)
                tb_writer.add_scalar("train/clip_fraction", agent.last_clip_fraction, gs)
                tb_writer.add_scalar("train/n_epochs_run", agent.last_n_epochs_run, gs)
                if not math.isnan(agent.last_explained_var):
                    tb_writer.add_scalar("train/explained_variance", agent.last_explained_var, gs)
                if bc_active and not math.isnan(agent.last_bc_loss):
                    tb_writer.add_scalar("train/bc_loss", agent.last_bc_loss, gs)
                    tb_writer.add_scalar("train/bc_coef", agent.last_bc_coef, gs)

            with metrics_path.open("a", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(_row_from_metrics(train_episodes_done, gs, epoch_idx, "train", m))

            if train_episodes_done % log_interval == 0:
                ruin_tag = " RUIN" if ep_ruin else ""
                print(f"[ep {train_episodes_done:04d}] step={gs}  "
                      f"pi_loss={agent.last_policy_loss:+.4g} v_loss={agent.last_value_loss:.4g} "
                      f"H={agent.last_entropy:.3f} KL={agent.last_approx_kl:+.4f} "
                      f"clip={agent.last_clip_fraction:.2f} EV={agent.last_explained_var:+.2f}  "
                      f"epochs={agent.last_n_epochs_run} eq=${m['final_equity']:.0f} "
                      f"trades={m['trades']}{ruin_tag}")

            rollout = Rollout()
            ep_pnl_log = []
            ep_equity = []
            ep_ruin = False
            ep_bar_idx = 0

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
                    (run_dir / "best_info.json").write_text(json.dumps({
                        "episode": train_episodes_done,
                        "epoch": epoch_idx,
                        "eval_count": eval_count,
                        "global_step": gs,
                        "metric": best_metric,
                        f"val_{best_metric}": best_value,
                        "val_trades": int(vm["trades"]),
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
            if bc_active:
                try:
                    _, cur_day_df = next(iter_sessions(feat, [train_env.current_date]))
                    cur_expert = compute_expert_actions(cur_day_df, action_space_list,
                                                        h=bc_h, noise_threshold=bc_thresh)
                except StopIteration:
                    cur_expert = None
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
        "algorithm": "ppo",
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
            "reward_normalize": use_norm,
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
        "algo": "ppo",
        "run_name": str(cfg["run"]["run_name"]),
        "seed": int(cfg["train"]["seed"]),
        "lr": float(ppo_cfg["lr"]),
        "clip_range": float(ppo_cfg["clip_range"]),
        "n_epochs": int(ppo_cfg["n_epochs"]),
        "minibatch_size": int(ppo_cfg["minibatch_size"]),
        "entropy_coef": float(ppo_cfg["entropy_coef"]),
        "gae_lambda": float(ppo_cfg["gae_lambda"]),
        "hidden_sizes": str(list(ppo_cfg["hidden_sizes"])),
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
    train_ppo(cfg)
