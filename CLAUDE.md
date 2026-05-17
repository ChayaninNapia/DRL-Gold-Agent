# CLAUDE.md — drl_intraday

## Project

**drl_intraday** — Intraday DRL trading agent for XAUUSD CFD on M1 bars.

**Canonical spec:** `proposal/PROPOSAL.md` (living document — the single source of truth for MDP/reward/experiments). `proposal/DRL_proposal_6509_6571_OLD_presented.pdf` is the **historical** version presented at proposal time; it is return-based and superseded — do NOT use it as reference. Project for Chayanin Napia 6509, Pronpipath Neamnoi 6571 — FRA 503 Deep Reinforcement Learning 2026. Implementation order: **Double DQN first, then A2C, then PPO** — all under the same MDP and trading environment.

- **Asset:** XAUUSD spot gold CFD
- **Timeframe:** 1-minute bars
- **Episode = 1 trading day** (all M1 bars per calendar date; no session-time filter). "session" and "day" are used interchangeably in code/logs.
- **Goal:** profitable intraday agent with positive Sharpe / Sortino under realistic transaction costs

## Python Environment

- **venv:** `d:\EA\.venv\` — always use this, NEVER system Python (cp874 encoding errors on Thai locale)
- **Run scripts:** `d:\EA\.venv\Scripts\python.exe <script>`
- **Key packages:** PyTorch 2.5.1+cu121, gymnasium 1.1.1, tensorboard 2.20.0, pandas 2.3.3, pyarrow 21.0.0, pyyaml 6.0.3
- **Ask before installing** new packages

## Workspace Layout

```
d:\EA\
  CLAUDE.md             # workspace context + experiment journal
  config.yaml           # single config for current experiment
  proposal/             # course proposal PDF
  src/                  # core code (importable modules)
    data.py             # load parquet, per-day split (absolute counts: n_train/n_val/n_test), variable-length per-day iterator
    cv.py               # 5-fold expanding-window inner CV on train split; aggregate_fold_scores (mean - 0.5*std)
    features.py         # market features (10): 5 close returns + MACD + STO + RSI + ATR + spread_pts
    env.py              # gymnasium portfolio env (PROPOSAL.md Sec.3): 16-dim state (10 market + 5 positional + equity_ratio), $10k capital, fixed 0.01 lot, spread cost, equity<=0 ruin, R1/R2/R4 reward modes + RunningStd reward normalizer
    ddqn.py             # Double DQN agent (QNet, ReplayBuffer, EpsilonSchedule, DDQNAgent)
    train.py            # DDQN training loop: DayCycleEnv (with shared RunningStd), pooled metrics + portfolio metrics, eval, train_ddqn() entry
    a2c.py              # A2C agent (ActorCritic shared trunk + 2 heads, GAE, Rollout, A2CAgent)
    train_a2c.py        # A2C training loop: episode-aligned rollouts, reuses DayCycleEnv + eval + metrics; tracks pnl_log/equity per-bar for portfolio metrics
    ppo.py              # PPO agent (reuses ActorCritic + Rollout + GAE from a2c.py; clipped surrogate, K-epoch minibatch SGD)
    train_ppo.py        # PPO training loop: same shape as train_a2c.py with PPO-specific TB diagnostics
    baselines.py        # 5 baseline strategies: FlatBaseline, LongBaseline, ShortBaseline, RandomBaseline, MACrossoverBaseline; all drop-in for evaluate_policy_per_session()
    expert.py           # daily hindsight action labeler (h-bar lookahead + noise_threshold -> action_idx); used by B1 (class-weighted BC warm-start) in train_a2c/train_ppo
    hpo.py              # Optuna objective: per-algo search space (R4 mode adds reward_beta + reward_dd_thresh), inner-CV loop, aggregate_fold_scores; min_trades filter
  scripts/              # entry points (add src/ to sys.path)
    sanity.py           # quick 20k-timestep run to verify pipeline (DDQN)
    train_ddqn.py       # full DDQN training (uses config.yaml as-is)
    train_a2c.py        # full A2C training (uses config.yaml as-is)
    train_ppo.py        # full PPO training (uses config.yaml as-is)
    run_seeds.py        # multi-seed runner: loops ddqn/a2c/ppo over [42,1337,2026], writes seeds_summary.csv + seeds_aggregate.json
    run_baselines.py    # evaluates all 5 baselines on test (or val/both), writes runs/baselines/baselines_results.csv + .json
    run_exp0.py         # Experiment 0 sweep: 3 rewards x 3 algos x 3 seeds = 27 runs, no HPO; ranks rewards by mean rank on val
    run_exp05{a,b,c,d,e}_parallel.py  # anti-collapse phase drivers (parallel N=4 subprocess): 0.5a γ-sweep, 0.5b/c on-policy γ/4-knob, 0.5d plain BC warm-start, 0.5e (B1) class-weighted BC. All historical; 0.5a fixed DDQN, 0.5b/c/d/e characterized on-policy collapse.
    run_hpo.py          # Optuna HPO entry: --algo, TPESampler + HyperbandPruner, sqlite (resumable), writes runs/hpo/<algo>/<study>_best.json
    run_final.py        # pulls best HPO trial -> retrain 500k x seeds on full train -> test once -> runs/final_<algo>/
  runs/                 # one folder per run: metrics.csv, trades.csv, best.pt, summary.json; TB under runs/_tb/<run_name>/
  data/                 # datasets
    GOLD_M1_last750_trading_days_to_2026-05-15.parquet  # current; updated 2026-05-18 (last date 2026-05-15)
    SILVER_M1_last750_trading_days_to_2026-05-15.parquet  # added 2026-05-18
    EURUSD_M1_last750_trading_days_to_2026-05-15.parquet  # added 2026-05-18
  research_papers/      # PDFs organised by topic
  tools/trade_viewer/   # web app: parquet viewer + trade-history candlestick chart (2 tabs, dark theme)
```

## Data Facts

- Main dataset: `data/GOLD_M1_last750_trading_days_to_2026-05-15.parquet`
- 1,027,906 rows, columns: `time` (datetime64 UTC), `open`, `high`, `low`, `close`, `tick_volume`, `spread`, `real_volume`
- 750 trading days total (~2.9 years calendar: 2023-06-20 → 2026-05-15). We do **not** filter by session time — keep every M1 bar of each calendar date.
- Additional datasets (same 750-day window, all ending 2026-05-15): `SILVER_M1_last750_trading_days_to_2026-05-15.parquet` (1,026,260 rows, 2023-06-20 →), `EURUSD_M1_last750_trading_days_to_2026-05-15.parquet` (1,057,740 rows, 2023-07-11 →).
- **Full window = 750 days → train/val/test = 600/75/75** (absolute day counts, chronological; no overlap). Configured via `data.n_train/n_val/n_test` in config.yaml — sum must equal `window_days`.
- Episode length varies per day (~958–1379 M1 bars). Median 1379 (full session); short days are known holidays (Jan 2, Dec 24/26/31, Jul 4, Thanksgiving). Do not pad or truncate.

### Inner CV (HPO only; held-out val/test never used)

- 5-fold expanding window on train, equal val size per fold (Option A).
- fold k val ends at `train_dates[n_train - (n_folds-k)*val_size]`; first fold's inner-train is the longest prefix that leaves room for all 5 folds.
- Defaults from `cfg["cv"]`: `n_folds=5`, `val_size=24` (5×24=120 days, the last 120 of train become inner-val across folds; inner-train sizes [480, 504, 528, 552, 576]).
- Trial score = `mean(fold_scores) − agg_penalty × std(fold_scores)` (default `agg_penalty=0.5`). Penalizes hyperparameter configs that are unstable across regimes.
- HPO objective metric: `cfg["cv"]["objective_metric"]` (default `sortino`), with `min_trades` filter disqualifying degenerate "do-nothing" policies (default 50).

## MDP Design

> **Authority: [proposal/PROPOSAL.md](proposal/PROPOSAL.md) Sec. 3.** This is a
> summary; if anything here disagrees with PROPOSAL.md, PROPOSAL.md wins. The MDP
> is **portfolio-based** (real capital + lots), NOT the old return-only abstraction.

- **Episode** = 1 trading day; **step** = 1 M1 bar. Capital resets to fixed
  `C0 = $10,000` every episode (episodes independent — no cross-episode compounding).
- **Action** = `cfg["env"]["action_space"]`; default `[-1, 0, +1]` (short/flat/long).
  Exp 2 expands to 5/9 actions. **Fixed lot 0.01** (= 1 oz; 1 lot = 100 oz →
  $1 price move = $1 P&L per 0.01 lot). Agent chooses direction only (no lot sizing).
- **Cost** = spread only (commission $0), from the parquet `spread` column,
  charged in dollars on every position change. **Execution at next-bar open**
  when the target position changes; else marked to next close. EOD force-flat.
- **Ruin termination:** `equity ≤ 0` at any bar → episode ends, fixed clipped
  `reward = -1`. Single hard threshold; no broker margin/stop-out modelling.

### State — 16-dim flat vector (10 market + 5 positional + 1 portfolio)

10 market + 5 positional are unchanged from before; **+1 new**: `equity_ratio =
equity_t / C0`. Full derivation formulas (ret/MACD/STO/RSI/ATR/spread, TL/POS/PR/
DR/HT, equity_ratio) are in **PROPOSAL.md Sec. 3.5** — do not duplicate them here.

### Reward — 3 variants compared in Exp 0 (PROPOSAL.md Sec. 3.6 / 6)

All share the same execution / spread-cost / EOD-flat / ruin rule; differ only
in the per-bar scalar. `Δ_t = equity_{t+1} − equity_t` (after spread cost):

- **R1** log-return of equity: `log(equity_{t+1} / equity_t)` — control/baseline
- **R2** raw net dollar P&L: `Δ_t`
- **R4** P&L − drawdown penalty: `Δ_t − β·max(0, DD_t − dd_thresh)`

Reward normalization (running-std) was the original proposal (PROPOSAL.md
Sec. 6.4) but was **disproved by A/B test on 2026-05-16** — `RunningStd`
collapsed the Q-net to flat. Current default: `env.reward.normalize: false`
in `config.yaml`. Comparability across R1/R2/R4 is now obtained via
per-reward learning-rate overrides (`dqn.lr_per_reward` etc.) instead.
Evaluation is never normalized either way. R3 (vol/SD-penalized) was
considered and dropped (overlaps R4, noisier on M1).

### Training (proposal §4.6)

- **Algorithm:** Double DQN, implemented from scratch in PyTorch ([src/ddqn.py](src/ddqn.py)). Online + target Q-nets, ε-greedy exploration, hard target update every `target_update_interval` env steps, Huber (SmoothL1) loss, Adam optimizer with `eps=1.5e-4`, gradient clipping at `max_grad_norm`. A2C and PPO will follow under the same MDP/env.
- **Policy network:** MLP `16 → hidden_sizes → n_actions` with ReLU. State is flat 16-dim (portfolio MDP), no recurrent.
- **Epoch** = 1 full pass over the train set (`sessions_per_epoch = len(train_dates)`: 600 on the held-out train split that `train_*.py`/`run_seeds.py` use, or 480–576 inside HPO inner-CV folds). Reported in TB/CSV/summary as `train/epoch` but not used for control flow.
- **Sampling within an epoch:** train sessions are **shuffled** at the start of every epoch (no chronological order during training). This is standard for off-policy DQN where the replay buffer breaks temporal correlation anyway. Splits themselves (train/val/test) remain chronological so no future leakage.
- **Eval cadence:** every `eval_every_sessions` completed train episodes (current `config.yaml` value: 22 ≈ once every 2 epochs at 600-session train; was 11 pre-2026-05-15). Each eval is a deterministic rollout over **all** val sessions, concatenated into one pooled return sequence for DeepScalper-style metrics.
- **Best checkpoint:** highest pooled val `<best_metric>` seen so far. `best_metric` is configurable in `train.best_metric` ∈ {`total_return`, `sharpe`, `sortino`}; current default is `total_return`. Saved as `best.pt` (torch state dict with `online`, `target`, `optim` keys).
- **Early stop:** `early_stop_patience` consecutive evals with no val `<best_metric>` improvement.
- **Logging:** `metrics.csv` + `trades.csv` (CSV), `summary.json` (final), and TensorBoard scalars under these groups:
  - `train/*` — full 6 trading metrics (`total_return`, `mdd`, `trades`, `winrate`, `sharpe`, `sortino`) **per training episode**, plus `episode_reward`, `episode_length`, `episode_count`, `epoch`, `exploration_rate`, `loss`, `mean_q`. Metrics come from the actual ε-greedy rollout used for training (no separate greedy pass) — early-training noise from high ε is preserved on purpose.
  - `val/*` — same 6 trading metrics, per eval pass (every `eval_every_sessions` train episodes)
  - `test/*` — same 6 trading metrics, logged once at end of run

## Evaluation Metrics (DeepScalper §5.2 — pooled period-level)

All metrics are computed on a **pooled per-bar return sequence** spanning every session in the phase (val or test). Sessions are still run independently — the env resets each day and force-closes at EOD, so there is no overnight gap in equity — but the per-bar log returns of all days are concatenated into one sequence for the metric. **One eval pass produces one row of metrics**, not one row per day.

| Metric | How it's computed in code today |
|---|---|
| `total_return` | `exp(sum(pooled_returns)) − 1` |
| `sharpe` | `mean(daily_r) / std(daily_r) × √252` — daily log-returns (sum per bar per day), annualized |
| `sortino` | `mean(daily_r) / std(downside_daily_r) × √252` — downside = days with `daily_r < 0`, denominator is **std** (not variance) |
| `mdd` | `min((equity − peak) / peak)` where `equity = exp(cumsum(returns))`, continuous across days within the phase |
| `trades` | total count of contiguous non-zero position blocks across all sessions |
| `winrate` | total wins / total trades, pooled across sessions |
| `avg_trade_pnl` | mean completed-trade log return across all sessions |
| `turnover` | `sum(abs(diff(positions)))` across all bars |
| `avg_holding_time` | mean bars_held per completed trade |

Equity continuity scope: **within a single eval pass only.** It does NOT carry from train → eval → test. Train phase has no Sharpe/equity logging at all (train returns come from the stochastic ε-greedy policy and are not meaningful as a trading signal). Each val pass starts equity at 1.0 at its first bar.

Best checkpoint = epoch with highest **pooled val `<best_metric>`** (configurable via `train.best_metric`).

`metrics.csv` columns: `episode, global_step, epoch, phase, total_return, mdd, trades, winrate, sharpe, sortino, avg_trade_pnl, turnover, avg_holding_time`. Rows are written for every training episode (`phase=train`), every val eval pass (`phase=val`), and once at the end for test (`episode=BEST, phase=test`). Optimizer diagnostics (loss, exploration_rate, mean_q) are logged to TensorBoard only, not CSV.

`trades.csv` columns: `episode, phase, day, entry_time, exit_time, side, entry_price, exit_price, bars_held, pnl_log`. Captured for every val pass and once for test (with `episode=BEST`).

`summary.json`: best checkpoint info (`episode`, `epoch`, `eval_count`, `global_step`, `metric`, `val_<metric>`), training stats (total_episodes, total_epochs, total_timesteps, total_evals, sessions_per_epoch, eval_every_sessions, n_train/val/test_sessions), and test metrics.

## How to Run

Portfolio MDP (PROPOSAL.md Sec. 3) is implemented and smoke-tested across all three trainers + baselines. Pipeline ready for Exp 0 → 1 → 2 → 3.

```powershell
# sanity
& 'd:\EA\.venv\Scripts\python.exe' 'd:\EA\scripts\sanity.py'

# full DDQN / A2C / PPO training (single seed, uses config.yaml run_name)
& 'd:\EA\.venv\Scripts\python.exe' 'd:\EA\scripts\train_ddqn.py'
& 'd:\EA\.venv\Scripts\python.exe' 'd:\EA\scripts\train_a2c.py'
& 'd:\EA\.venv\Scripts\python.exe' 'd:\EA\scripts\train_ppo.py'

# multi-seed runner (seeds 42 1337 2026, writes seeds_summary.csv + seeds_aggregate.json)
& 'd:\EA\.venv\Scripts\python.exe' 'd:\EA\scripts\run_seeds.py' --algo ppo --run-name ppo_exp1
& 'd:\EA\.venv\Scripts\python.exe' 'd:\EA\scripts\run_seeds.py' --algo all  # runs all three algos

# baseline evaluation (writes runs/baselines/baselines_results.csv + .json)
& 'd:\EA\.venv\Scripts\python.exe' 'd:\EA\scripts\run_baselines.py'
& 'd:\EA\.venv\Scripts\python.exe' 'd:\EA\scripts\run_baselines.py' --split both --ma-fast 20 --ma-slow 60

# Experiment 0 — Reward Selection (3 rewards x 3 algos x 3 seeds = 27 runs, NO HPO).
# Winner = best mean rank across algos on val. Output: runs/exp0/exp0_summary.csv + exp0_winner.json
& 'd:\EA\.venv\Scripts\python.exe' 'd:\EA\scripts\run_exp0.py'
#   then update config.yaml env.reward.mode = <winner> before Exp 1.

# Experiment 1 (Aggressive HPO on the Exp-0 winning reward): HPO per algo, then final retrain + test
& 'd:\EA\.venv\Scripts\python.exe' 'd:\EA\scripts\run_hpo.py' --algo ddqn   # defaults: 12 trials, 3 folds, 100k/fold
& 'd:\EA\.venv\Scripts\python.exe' 'd:\EA\scripts\run_final.py' --algo ddqn  # retrain best trial 500k x 3 seeds -> test
#   (repeat --algo a2c / ppo for both scripts)

# TensorBoard (TB lives under runs/_tb/, separate from run artifacts)
& 'd:\EA\.venv\Scripts\tensorboard.exe' --logdir 'd:\EA\runs\_tb' --port 6006
```

## Conventions

- **Pooled period-level metrics** are canonical (DeepScalper §5.2). Sharpe/Sortino/MDD/TR/trades/winrate are computed over the concatenated return sequence of a whole phase (val or test), not per-day-then-averaged.
- **All config reads must use `encoding="utf-8"`** — venv runs on system Python 3.9 whose default `Path.read_text()` uses cp874 on Thai locale.
- **No subprojects / subfolders.** One project, one config, one set of `*.py`; new experiments = new `run_name` and a new entry in [JOURNAL.md](JOURNAL.md).
- **Episode length varies per day.** `TL_t` is computed from the last available bar of *that day*, not a fixed T.
- **No future leakage.** Execution price uses `open_{t+1}` whenever action changes; otherwise `close_t`. Any HTF feature (when added) must use a closed HTF bar with `shift(1)`.

---

## Insights (still-active findings from past experiments)

TL;DR of lessons learned. For full context including A/B numbers, dates, and rejected alternatives, open [JOURNAL.md](JOURNAL.md). Items here are only the ones that still constrain current decisions.

> ⚠️ **2026-05-15 — MDP rebuilt to portfolio-based (PROPOSAL.md).** Code is now
> portfolio-MDP-compliant (16-dim state, $10k capital, spread cost, R1/R2/R4
> reward with optional running-std normalization (default OFF after 2026-05-16
> A/B disproof — kept in code as a toggle), equity≤0 ruin termination). The
> rebuild covered `src/env.py`, all three trainers (`train.py`, `train_a2c.py`,
> `train_ppo.py`), `config.yaml`, `scripts/run_baselines.py`, `scripts/run_seeds.py`,
> `src/hpo.py` (R4 search-space), `scripts/run_final.py`, and added
> `scripts/run_exp0.py`. Agent code unchanged (obs_dim is parametric).
> Pipeline smoke-tested end-to-end on all three algos and on baselines.
>
> Many insights below were learned under the OLD return-based MDP. Treat
> anything referencing "commission cost", "equity starts at 1.0", "log-return
> reward", "15-dim state", or HPO/Aggressive results as **historical** — they
> must be re-validated under the new portfolio MDP. The reward-collapse /
> over-trading and on-policy sample-inefficiency findings are likely still
> directionally true but unproven on the new MDP. No real (non-smoke) runs
> have been done yet.

- **Normalization didn't help (2026-05-14).** Rolling z-score on market features (windows 60, 240) made things worse or no better than raw. Removed from `src/features.py`. Real bottleneck was over-trading collapsing to "do nothing", not feature scale — don't re-introduce normalization as a fix for poor trading metrics; look at exploration / commission / training duration instead.
- **DDQN was migrated SB3 → from-scratch PyTorch (2026-05-14).** A/B at 3 seeds × 150k steps passed at 1σ on policy-quality metrics (best val, test return, MDD, winrate). Sharpe/Sortino "FAIL" results were n=3 ratio-metric artifacts on tiny absolute values, not a real divergence. Don't bring SB3 back. All three algorithms are scratch implementations for fair comparison.
- **A2C and PPO need tighter gradients than DDQN.** A2C uses `lr=0.0007`, `max_grad_norm=0.5`; PPO uses `lr=0.0003`, `max_grad_norm=0.5`. DDQN uses `lr=0.0045`, `max_grad_norm=10.0`. On-policy gradients explode more readily; keep these gaps if you tune.
- **γ=0.30 is the DDQN project default (Exp 0.5a, 2026-05-16).** It solved DDQN collapse (0/3, +2.23% mean test_ret) and is set in `config.yaml dqn.gamma`. This benefit is **off-policy-specific** — do NOT assume A2C/PPO inherit it (Exp 0.5b: γ=0.3 → 3/3 collapse for both).
- **On-policy (A2C/PPO) collapse at γ=0.3 is NOT fixable by hyperparameter tuning, BC warm-start, NOR class-weighted BC — the imitation branch is exhausted (Exp 0.5b/c/d + B1/0.5e, 2026-05-16).** γ, gae_lambda, value_coef, entropy_coef, n_epochs all tried (1b/1c) → 3/3 collapse. Plain BC warm-start from a hindsight expert (1d) → 3/3 collapse: BC kills the uniform-random collapse but the policy commits to **flat** (expert is **85.7% flat** at h=5/th=0.0005 — measured exactly this session, worse than the earlier ~63–70% guess). B1/0.5e added inverse-frequency class-weighted CE so flat can't be the easy answer: it **worked** (runs traded 3000–6000×/eval during the BC phase) but **still collapsed 6/6 to val_trades=0 once BC annealed out** — proving flat-commitment was a *symptom*, not the root cause. Net: every on-policy fix ends at val_trades≈0. Root cause: **the post-BC RL phase has no learning signal that makes trading positive-EV** (credit-assignment), worsened by `rl_lr` annealing to the tiny R4-scale 1e-5/1e-6. Ruled out this session: over-training (300k = only 0.38 epoch) and train→val regime mismatch (DDQN+R4 γ=0.30 made +2.23% test on the *identical* split, 0/3 collapse — only the algo class differs). Do NOT try more imitation/BC variants. Remaining: VC-PPO decoupled critic λ, switch algo (SAC-Discrete / Recurrent PPO per `proposal/algorithm_extensions.md` — deletion currently staged, recover via `git restore --staged --worktree proposal/algorithm_extensions.md`), or **proceed Exp 1 DDQN-only and document A2C/PPO as a characterized structural negative (recommended)**. DDQN (off-policy) is unaffected and remains the working baseline at γ=0.30.
- **PPO advantage normalization is per-minibatch, A2C is per-rollout.** Don't unify them — PPO's K-epoch SGD makes rollout-level statistics go stale; A2C's single update means rollout-level is fine.
- **Episode-aligned rollouts for A2C/PPO.** 1 rollout = 1 trading day. `last_value = 0` at episode end (env force-flattens at EOD, P&L final). Do not bootstrap partial trajectories.
- **Train-phase Sharpe is meaningless and not logged.** Train returns come from a stochastic ε-greedy (DDQN) or sampled (A2C/PPO) policy. DeepScalper itself only reports test. Don't add train Sharpe/Sortino back.
- **Sortino bug — denominator is `std`, not `var` (2026-05-15 fix).** If you ever rewrite the metric, make sure `downside.std()` is used. Old code reported variance and produced wildly wrong numbers.
- **Sharpe/Sortino annualization is daily-aggregated × √252.** When `day_lengths` is available (val/test), per-bar returns are summed per day before computing the ratio. Per-bar fallback is only for single-episode train logging and is not annualized.
- **`runs/` wiped clean 2026-05-15 (twice).** Final wipe: all return-based HPO/runs deleted as invalid after the portfolio-MDP redesign. Only `.gitkeep` remains. Nothing to compare against until the rebuilt env runs Exp 0.
- **TensorBoard logs live under `runs/_tb/<run_name>/`, NOT `runs/<run_name>/tb/`** (2026-05-15). Event files are separated from run artifacts (best.pt, csv, json). Always `tensorboard --logdir runs/_tb`. Every run also logs `add_hparams()` (algo, lr, seed, hidden_sizes, … → val_<best_metric> + test metrics) and `add_text("config")` — use the HPARAMS tab to compare HPO trials.
- **HPO adapter contract — `cfg["_hpo"]` is a runtime-only injection.** When present, `train_*()` use the CV fold's `inner_train_dates`/`inner_val_dates` instead of the held-out split, honor `timesteps_override`, and **early-return before the test rollout** with `{hpo_objective, best_metric, val_trades}`. `scripts/train_*.py`, `run_seeds.py`, `run_baselines.py` never set `_hpo` so they are unaffected. `_hpo` is stripped before any `yaml.safe_dump` (it holds pandas Timestamps). Don't persist `_hpo` or let it reach yaml.
- **HPO timesteps must be large enough for ≥1 val eval per fold.** If training ends before the first eval fires, `best_value` stays −inf, the trial returns −inf and is filtered. 100k/fold with the current `eval_every_sessions=22` is comfortably safe (on 480–576 inner-train sessions per fold); do not cut per-fold budget so low that no eval fires.
- **Experiment structure is now Exp 0→1→2→3** (PROPOSAL.md Sec.5): Exp 0 = reward selection (R1/R2/R4 × 3 algos × 3 seeds, **no HPO**, pick by mean rank across algos), then Exp 1 = algo comparison with full HPO on the winning reward, Exp 2 = action space, Exp 3 = LSTM/GRU. The old "Aggressive HPO Experiment 1" runs (return-based) are **invalid and deleted** — the Aggressive 3-fold/12-trial/100k HPO recipe itself is still the intended HPO config, just to be re-run on the rebuilt portfolio MDP starting from the Exp-0 winning reward.

For history (return-based bake-off, old Aggressive HPO results, portfolio-MDP decision log) see [JOURNAL.md](JOURNAL.md). Active spec is always [proposal/PROPOSAL.md](proposal/PROPOSAL.md).
