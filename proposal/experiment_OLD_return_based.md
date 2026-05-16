# Experimental Setup

## Objective

This study evaluates deep reinforcement learning (DRL) methods for intraday XAUUSD trading using one-minute (M1) market data. The experiments are designed to answer three main questions:

1. Which DRL algorithm learns the trading task most effectively under the same market environment?
2. Does expanding the discrete action space improve policy quality or make learning harder?
3. Does adding sequential function approximation, such as LSTM or GRU, improve performance over a standard MLP policy/value network?

The final selected model will be compared against simple baseline trading strategies to assess whether the learned policy provides meaningful improvement beyond naive or rule-based behavior.

## Dataset and Market Environment

The dataset consists of XAUUSD M1 bars stored in `data/GOLD_M1_last750_trading_days_to_2026-05-01.parquet`. The current setup uses the most recent 750 trading days (1,027,754 M1 bars), spanning approximately 2.9 calendar years from 2023-06-05 to 2026-04-30. The larger window was adopted on 2026-05-15 to support nested cross-validation for hyperparameter selection (see the train/validation/test split below).

Each trading day is treated as one independent episode. Within an episode, each environment step corresponds to one M1 bar. Episode length may vary depending on the number of available bars for that day. The environment does not pad or truncate daily sessions.

The time-series split is performed by trading day using absolute day counts (not ratios, to avoid rounding ambiguity). The sum of the three splits must equal the full window of 750 days:

| Split | Sessions | Date Range | Usage |
|---|---:|---|---|
| Train | 600 days | 2023-06-05 -> 2025-09-29 | Policy learning and hyperparameter selection (via inner CV) |
| Validation | 75 days | 2025-09-30 -> 2026-01-14 | Held-out model selection and early stopping |
| Test | 75 days | 2026-01-15 -> 2026-04-30 | Final out-of-sample reporting |

The split must preserve chronological order:

```text
Train period -> Validation period -> Test period
```

No validation or test data may be used for training, hyperparameter selection, normalization fitting, or checkpoint selection.

### Inner Cross-Validation for Hyperparameter Selection

Hyperparameter optimization operates on the train split only; the held-out validation and test splits are never touched during HPO. The protocol is a 5-fold expanding-window inner CV with equal validation size per fold:

- 5 folds, 24 validation days per fold (5 × 24 = the last 120 train days become inner-validation across folds).
- Fold *k*'s inner-validation ends at `train_dates[600 − (5 − k) × 24]`; the first fold's inner-train is the longest prefix that still leaves room for all 5 folds. Inner-train sizes are `[480, 504, 528, 552, 576]` days.
- Trial score = `mean(fold_scores) − 0.5 × std(fold_scores)`, penalizing hyperparameter configurations that are unstable across market regimes.

This keeps the held-out 75-day validation split untouched during HPO so it can serve as a clean checkpoint-selection signal during the final retrain, and keeps the 75-day test split for a single final out-of-sample measurement per algorithm.

## Episode Ordering and Shuffling

Training may shuffle the order of daily episodes inside the training split. This is acceptable because one episode equals one full trading day, and the agent is reset at the beginning of each day. Shuffling training days can reduce overfitting to a particular market-regime order.

However, the timestep order inside each day must never be shuffled. The within-day sequence must remain chronological because state transitions, execution prices, rewards, and position holding periods depend on the true temporal order.

Allowed:

```text
Training episode order: Day 20 -> Day 3 -> Day 71 -> Day 12
```

Not allowed:

```text
Within one day: t10 -> t3 -> t25 -> t1
```

Validation and test evaluation should be run in chronological order to reflect realistic out-of-sample deployment.

## State Representation

The baseline state is a flat feature vector consisting of market features and positional features.

Market features include return windows and technical indicators. The 10 market features are:

- 5 close-to-close return windows over `[1, 5, 15, 30, 60]` M1 bars: `r_{t-w,t} = (C_t − C_{t-w}) / C_{t-w}`
- MACD raw value: `EMA12(C) − EMA26(C)` (no signal line or histogram)
- Stochastic Oscillator over 14 bars
- RSI over 14 bars
- ATR over 14 bars
- Broker spread in points: `spread_pts` column from the parquet data

Positional features describe the agent's current trading state, including:

| Feature | Meaning |
|---|---|
| `tl` | Bars remaining until the forced end-of-session close |
| `pos` | Current position scalar: −1, 0, or +1 (or partial sizes in Experiment 2) |
| `pr` | Unrealized log return since entry, net of entry transaction cost |
| `dr` | Cumulative log return for the current session so far |
| `ht` | Bars held in the current position (0 when flat) |

For Experiments 1 and 2, the function approximator is an MLP. For Experiment 3, sequential models will replace or augment the MLP with LSTM and GRU architectures.

## Action Space

The current baseline action space is discrete and uses target-position semantics:

| Action Index | Target Position | Meaning |
|---:|---:|---|
| 0 | -1 | Short |
| 1 | 0 | Flat |
| 2 | +1 | Long |

At each step, the agent chooses the next target position. If the target position differs from the current position, the trade is executed using next-bar open execution. At the end of each trading day, the environment forces the position to flat so that no position is carried overnight.

Experiment 2 will expand the action space to include finer position sizes. A candidate expanded action space is:

```text
[-1.0, -0.5, 0.0, +0.5, +1.0]
```

This represents full short, half short, flat, half long, and full long. If the implementation supports a larger action space, an additional setup may be tested:

```text
[-1.0, -0.75, -0.5, -0.25, 0.0, +0.25, +0.5, +0.75, +1.0]
```

The same transaction cost model must be applied consistently across all action-space settings.

## Reward and Execution Model

The reward is the per-step log return after transaction cost. When the agent changes position, execution occurs at the next bar open. When the position is unchanged, the return is marked to the next bar close. At the end of the session, the environment forces the position to flat at the current bar close and charges the corresponding transaction cost.

Transaction cost is charged according to the absolute change in position:

```text
cost = commission * execution_price * abs(delta_position)
```

The current commission parameter is:

```text
commission = 0.00005
```

The daily log return is the sum of per-bar log returns within one session. The reported total return for a multi-day period is the arithmetic cumulative return:

```text
total_return = exp(sum of all per-bar log returns over the period) - 1
```

## Common Controlled Settings

To make the comparison fair, the following settings should remain fixed unless they are the explicit variable under investigation:

| Component | Controlled Setting |
|---|---|
| Dataset | Same XAUUSD M1 file (`GOLD_M1_last750_trading_days_to_2026-05-01.parquet`) |
| Window | Same 750 trading-day window |
| Train/validation/test split | Same chronological 600 / 75 / 75 split |
| Episode definition | One trading day per episode |
| Reward function | Same log-return reward after cost |
| Execution model | Same next-bar open execution and end-of-day flattening |
| Feature set | Same features unless testing sequence input |
| Evaluation policy | Deterministic action selection |
| Best checkpoint selection | Validation metric only (`total_return`, `sharpe`, or `sortino`) |
| Final reporting | Test set only after model selection |

Because DRL training has high variance, each experimental condition should be repeated with multiple random seeds, for example:

```text
seeds = [42, 1337, 2026]
```

If compute budget allows, using 5-10 seeds would make the results more reliable. Results should be reported as mean and standard deviation across seeds.

## Experiment 1: Algorithm Comparison with MLP

### Purpose

The first experiment compares DDQN, A2C, and PPO under the same MDP, action space, feature set, reward function, transaction cost, and train/validation/test split.

### Setup

| Item | Setting |
|---|---|
| Algorithms | DDQN, A2C, PPO |
| Function approximator | MLP |
| Action space | `[-1, 0, +1]` |
| Input type | Flat feature vector |
| Episode | One trading day |
| Training order | Training days may be shuffled (DDQN: replay buffer; A2C/PPO: 1 rollout = 1 trading day) |
| Validation/test order | Chronological |
| Best model selection | Validation metric |

### Hypothesis

PPO and A2C may learn more stable policies because they directly optimize a stochastic policy and can handle noisy rewards, while DDQN may be more sensitive to exploration and action-value overestimation even with Double DQN. However, DDQN may still perform well because the action space is small and discrete.

### Output

The best algorithm is selected using validation performance, primarily total return or a risk-adjusted metric such as Sharpe or Sortino. The selected model from this experiment becomes the reference point for Experiment 2.

## Experiment 2: Action Space Expansion

### Purpose

The second experiment evaluates whether a larger discrete action space improves performance by allowing the agent to control position size more precisely.

### Setup

| Item | Setting |
|---|---|
| Algorithms | DDQN, A2C, PPO |
| Function approximator | MLP |
| Baseline action space | `[-1, 0, +1]` |
| Expanded action space | `[-1, -0.5, 0, +0.5, +1]` |
| Optional larger action space | `[-1, -0.75, -0.5, -0.25, 0, +0.25, +0.5, +0.75, +1]` |
| Other settings | Same as Experiment 1 |

### Hypothesis

An expanded action space may improve performance by allowing partial exposure and smoother risk control. However, it also increases exploration difficulty. DDQN may be affected more strongly by the larger number of discrete actions, while PPO and A2C may adapt more smoothly through their policy distributions.

### Output

This experiment identifies the best combination of algorithm and action space. The winner is used as the base method for Experiment 3.

## Experiment 3: Sequential Function Approximation

### Purpose

The third experiment tests whether sequence-aware models improve trading performance by capturing temporal patterns that are not fully represented in the flat MLP state.

### Setup

| Item | Setting |
|---|---|
| Base algorithm | Best algorithm from Experiment 2 |
| Base action space | Best action space from Experiment 2 |
| Function approximators | MLP, LSTM, GRU |
| Input type | Rolling sequence of observations |
| Sequence length | To be fixed before training, e.g. 30, 60, or 120 M1 bars |
| Split | Same chronological train/validation/test split |

### Sequence Construction

For LSTM and GRU models, each state should contain only current and past observations. No future bars may be included in the sequence window. At the beginning of each day, the sequence can be handled using either zero-padding, repeated first observation padding, or a warm-up period. The chosen method must be applied consistently across all sequence models.

### Hypothesis

LSTM and GRU models may improve performance if useful market structure depends on recent temporal dynamics beyond the handcrafted indicators. However, recurrent models may also overfit because the dataset is relatively small, especially when the number of parameters increases.

### Output

The experiment compares MLP, LSTM, and GRU versions of the best DRL setup. The final model is selected based on validation performance and then evaluated once on the test set.

## Final Baseline Comparison

After selecting the best DRL model, it will be compared against simple baseline strategies on the same test period.

Recommended baselines:

| Baseline | Description |
|---|---|
| Flat-only | Always hold no position (zero-trade baseline) |
| Long-only | Enter long at session open, flatten at session close |
| Short-only | Enter short at session open, flatten at session close |
| Random policy | Uniformly random action each bar, same action space as the DRL agent |
| Moving average crossover | Rule-based long/short/flat strategy using two fixed MA windows |

All baselines must use the same execution model, transaction cost, and end-of-day flattening rule as the DRL agents.

## Evaluation Metrics

The following metrics should be reported for validation and test periods:

| Metric | Unit / Interpretation |
|---|---|
| Total return | Arithmetic cumulative return: `exp(sum of per-bar log returns) − 1`, reported as a decimal (e.g. 0.05 = 5%) |
| Maximum drawdown | `min((equity − peak) / peak)` where `equity = exp(cumsum(returns))`, computed continuously across all sessions in the phase |
| Sharpe ratio | Annualized daily Sharpe: `mean(daily_r) / std(daily_r) × √252`, where `daily_r` is the sum of per-bar log returns within each trading day |
| Sortino ratio | Annualized daily Sortino: `mean(daily_r) / std(downside_daily_r) × √252`, where downside is days with `daily_r < 0` and the denominator is standard deviation (not variance) |
| Number of trades | Count of completed trades (each contiguous non-zero position block is one trade) |
| Win rate | Percentage of profitable completed trades |
| Average trade PnL | Mean completed-trade log return |
| Turnover | Sum of absolute position changes |
| Average holding time | Average bars held per trade |

Sharpe and Sortino are computed on the daily return series (one value per trading day = sum of that day's per-bar log returns) and annualized by multiplying by √252. The annualization factor 252 assumes a standard trading calendar with 252 trading days per year.

## Leakage and Validity Checks

The following checks are required to avoid misleading results:

1. Do not shuffle the chronological train/validation/test split.
2. Do not fit normalization statistics using validation or test data.
3. Do not select hyperparameters or checkpoints using test-set performance.
4. Do not shuffle timesteps inside a trading day.
5. Ensure rolling indicators use only current and past data.
6. Ensure LSTM/GRU sequence windows do not include future bars.
7. Apply transaction costs to all DRL methods and all baselines.
8. Use the same random seeds across comparable algorithms when possible.
9. Report mean and standard deviation across seeds.
10. Keep training budget comparable across algorithms.

## Reporting Plan

Each experiment should produce a table with one row per condition and columns for validation and test metrics. The main comparison should use validation results for model selection and test results only for final reporting.

Recommended table format:

| Experiment | Algorithm | Action Space | Model | Seed | Val Return | Val Sharpe | Val MDD | Test Return | Test Sharpe | Test MDD | Trades |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Exp. 1 | DDQN | 3 actions | MLP | 42 | | | | | | | |
| Exp. 1 | A2C | 3 actions | MLP | 42 | | | | | | | |
| Exp. 1 | PPO | 3 actions | MLP | 42 | | | | | | | |

The final discussion should explain not only which method achieved the highest return, but also whether the result is robust across seeds, whether it trades excessively, whether it suffers large drawdowns, and whether it remains better than simple baselines after transaction costs.
