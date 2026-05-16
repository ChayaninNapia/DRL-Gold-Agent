# PROPOSAL — drl_intraday (canonical)

> **This file is the single source of truth** for the MDP, reward, and experiment
> design of this project. Code, configs, CLAUDE.md, and JOURNAL.md must conform to
> this document. It is a *living document* — update it here first, then propagate.
>
> `proposal/DRL_proposal_6509_6571_OLD_presented.pdf` is the version presented at
> proposal time. It is **return-based and superseded**; kept only as a historical
> record. Do not use it as a reference for implementation.
>
> Project: Chayanin Napia (6509), Pronpipath Neamnoi (6571) — FRA 503 Deep
> Reinforcement Learning, 2026.

---

## 1. Overview & Goal

A deep-RL agent that trades **XAUUSD spot gold CFD intraday on 1-minute (M1)
bars**, simulated as a **portfolio with real capital and lot-based position
sizing**.

- One episode = one trading day (all M1 bars of a calendar date; no session-time
  filter). "session" and "day" are interchangeable in code/logs.
- Goal: a profitable intraday agent with **positive Sharpe / Sortino under
  realistic spread cost**, that beats naive and rule-based baselines after costs.

**Key change vs the old PDF proposal:** the environment is no longer a
return-only abstraction. It now simulates a **dollar-denominated portfolio**:
fixed starting capital, fixed lot size, dollar P&L, and episode termination on
ruin. This makes drawdown, capital preservation, and ruin first-class.

---

## 2. Dataset & Split

- **Dataset:** `data/GOLD_M1_last750_trading_days_to_2026-05-01.parquet` —
  1,027,754 M1 bars, 750 trading days, 2023-06-05 → 2026-04-30.
  Columns: `time` (UTC), `open`, `high`, `low`, `close`, `tick_volume`,
  `spread`, `real_volume`. Every M1 bar of each date is kept (no padding/truncation;
  episode length varies ~958–1379 bars).
- **Split (chronological, absolute day counts, no overlap):**
  - Train 600 days (2023-06-05 → 2025-09-29)
  - Validation 75 days (2025-09-30 → 2026-01-14)
  - Test 75 days (2026-01-15 → 2026-04-30)
- **Inner cross-validation for HPO** (train split only; held-out val/test never
  touched during HPO):
  - Primary ("Aggressive", compute-constrained): **3-fold** expanding window,
    24 val days/fold.
  - Optional (if compute allows): 5-fold, 24 val days/fold.
  - Trial score = `mean(fold_scores) − 0.5·std(fold_scores)` (rewards
    cross-regime stability).
  - A trial whose best inner-val checkpoint trades fewer than `min_trades`
    (default 50, averaged across folds) is disqualified (`-inf`) to filter
    degenerate do-nothing policies.

---

## 3. Portfolio MDP

### 3.1 Episode & capital

- Episode = 1 trading day. **Starting capital is reset to a fixed
  `C0 = $10,000` at the start of every episode** (episodes are independent — no
  cross-episode compounding, so CV/splitting stays valid).
- `equity_t` = cash + unrealized P&L of the open position, marked to the current
  bar. `equity_0 = C0`.

### 3.2 Action & position sizing

- **Action space (Exp 0/1):** discrete `{short, flat, long}` = target position
  sign ∈ {−1, 0, +1}.
- **Fixed lot size `L = 0.01`** (broker minimum for XAUUSD). The agent does NOT
  choose lot size in Exp 0/1 (action only sets direction). Dynamic/variable lot
  sizing is out of scope (see Sec. 8).
- **Contract:** 1 standard lot = 100 troy oz → `0.01 lot = 1 oz`. A `$1`
  move in gold price = **`$1` P&L per 0.01 lot** held.
- Exp 2 expands the *direction/size* action set
  (`[-1,-0.5,0,+0.5,+1]`, then 9-action) — still mapped through the same
  contract math.

### 3.3 Execution & cost

- **Execution at next-bar open** when the target position differs from the
  current one; otherwise marked to the next bar close. Unchanged from the prior
  design (no look-ahead: state at `t` uses only data ≤ `t`).
- **Transaction cost = spread only.** Commission = **$0**. The spread cost of a
  position change is taken from the `spread` column of the parquet at the
  execution bar, converted to dollars via the contract multiplier
  (`spread_points × point_value × lot`). Spread remains available as a state
  feature *and* is now charged as real cost on every position change.
- **End of day:** the env force-flattens the position at the last bar's close
  (spread cost applied to the closing trade).

### 3.4 Termination (ruin)

- If `equity_t ≤ 0` at any bar → **episode terminates immediately** with a
  fixed clipped penalty `reward = -1`. No broker-style margin-level / stop-out
  modelling — a single ruin threshold (`equity ≤ 0`) is the only hard
  constraint (a CMDP-style termination constraint).
- Normal episodes terminate at the last bar of the day (EOD force-flat).

### 3.5 State (16-dim flat vector)

The state at bar $t$ is

$$
s_t = \big[\, \underbrace{m_t}_{\text{market (10)}} \;,\; \underbrace{p_t}_{\text{positional (5)}} \;,\; \underbrace{e_t}_{\text{portfolio (1)}} \,\big] \in \mathbb{R}^{16}
$$

no recurrence (flat MLP input). All features are causal — they use only data at
or before bar $t$. Let $C_t, H_t, L_t$ be the close/high/low at bar $t$.

#### Market features $m_t$ (10) — formulas as implemented in `src/features.py`

**1–5. Close-to-close returns** over windows $w \in \{1,5,15,30,60\}$ bars:

$$
\text{ret}_w(t) = \frac{C_t - C_{t-w}}{C_{t-w}}
$$

**6. MACD** — difference of fast/slow EMAs of the close ($\text{EMA}_n$ with
span $n$, `adjust=False`):

$$
\text{MACD}_t = \text{EMA}_{12}(C)_t - \text{EMA}_{26}(C)_t
$$

**7. Stochastic Oscillator** over $n = 14$ bars:

$$
\text{STO}_t = 100 \cdot \frac{C_t - \min_{i \in [t-n+1,\,t]} L_i}{\max_{i \in [t-n+1,\,t]} H_i - \min_{i \in [t-n+1,\,t]} L_i}
$$

**8. RSI** over $n = 14$ bars. With $\Delta_t = C_t - C_{t-1}$,
$\text{gain}_t = \max(\Delta_t, 0)$, $\text{loss}_t = \max(-\Delta_t, 0)$, and
$\overline{G}_t, \overline{L}_t$ the $n$-bar simple means of gain/loss:

$$
\text{RS}_t = \frac{\overline{G}_t}{\overline{L}_t}, \qquad
\text{RSI}_t = 100 - \frac{100}{1 + \text{RS}_t}
$$

**9. ATR** over $n = 14$ bars. With true range
$\text{TR}_t = \max\big(\,H_t - L_t,\; |H_t - C_{t-1}|,\; |L_t - C_{t-1}|\,\big)$:

$$
\text{ATR}_t = \frac{1}{n}\sum_{i = t-n+1}^{t} \text{TR}_i
$$

**10. Spread** in broker points, taken directly from the `spread` column:

$$
\text{SP}_t = \text{spread\_points}_t
$$

Warm-up NaNs (first $n$ bars) are filled with $0$ so the env can use bar 0.

#### Positional features $p_t$ (5) — as implemented in `src/env.py`

Let the position held entering bar $t$ be $a_t \in \{-1,0,+1\}$ (Exp 0/1),
opened at bar index $t_e$ with entry execution price $p_e$ and entry cost
$\kappa_e$ (dollars). Let $\eta = \text{contract\_size} \times \text{lot}$
denote dollars-per-price-unit (with the defaults $100 \times 0.01 = 1$, so a
$\$1$ gold move equals $\$1$ P&L per 0.01 lot). $N$ = number of bars in the day.

**1. Time-to-close** (bars until the forced EOD flatten):

$$
\text{TL}_t = (N - 1) - t
$$

**2. Position scalar:**

$$
\text{POS}_t = a_t
$$

**3. Unrealized return since entry**, net of entry cost (0 when flat). The
numerator is in *price units* so the entry cost (in dollars) is divided by
$\eta$ to make the expression dimensionally consistent:

$$
\text{PR}_t = \begin{cases}
\dfrac{a_t\,(C_t - p_e) \;-\; \kappa_e/\eta}{p_e} & a_t \neq 0 \\[2mm]
0 & a_t = 0
\end{cases}
$$

With the default $\eta = 1$ the formula reduces to $(a_t(C_t - p_e) - \kappa_e)/p_e$;
the explicit $\eta$ matters only if `contract_size` or `lot` changes
(e.g. a future variable-sizing experiment, see Sec. 8).

**4. Cumulative session return** so far — the running sum of per-bar log
returns since the episode start:

$$
\text{DR}_t = \sum_{i=0}^{t-1} r_i
$$

**5. Holding time** in bars (0 when flat):

$$
\text{HT}_t = \begin{cases} t - t_e & a_t \neq 0 \\ 0 & a_t = 0 \end{cases}
$$

#### Portfolio feature $e_t$ (1, new)

$$
\text{equity\_ratio}_t = \frac{\text{equity}_t}{C_0}
$$

where $C_0 = \$10{,}000$ is the fixed starting capital. This is the only new
state dimension vs the prior 15-dim design; the agent needs it to behave
sensibly under the $\text{equity} \le 0$ termination (Sec. 3.4).

### 3.6 Reward

All three reward variants share the same execution model, spread cost, EOD
flatten, and ruin rule (Sec. 3.3–3.4). They differ only in the per-bar scalar fed to
the optimizer. **Reward normalization (Sec. 6.4) is applied to every variant** so
the three are comparable at a single learning rate.

Let the per-bar dollar change of equity after spread cost be

$$
\Delta_t = \text{equity}_{t+1} - \text{equity}_t
$$

The three reward variants (before normalization):

$$
\begin{aligned}
\textbf{R1 (log-return):}\quad & r_t = \log\!\left(\dfrac{\text{equity}_{t+1}}{\text{equity}_t}\right) \\[2mm]
\textbf{R2 (dollar P\&L):}\quad & r_t = \Delta_t \\[2mm]
\textbf{R4 (P\&L − DD penalty):}\quad & r_t = \Delta_t \;-\; \beta \cdot \max\!\big(0,\; \text{DD}_t - \tau\big)
\end{aligned}
$$

where the in-episode drawdown is

$$
\text{DD}_t = \frac{\text{peak\_equity}_t - \text{equity}_t}{\text{peak\_equity}_t},
\qquad
\text{peak\_equity}_t = \max_{0 \le i \le t}\, \text{equity}_i
$$

- $\beta$ (penalty weight) and $\tau = \texttt{dd\_thresh}$ (drawdown tolerance)
  are R4 hyperparameters, tuned only in Exp 1 for the winning reward — **not** in
  Exp 0 (see Sec. 5).
- On a ruin step ($\text{equity}_t \le 0$) every variant emits the fixed clipped
  value $r_t = -1$ and the episode ends.

(R3 mean-SD / volatility-penalized reward was considered and **dropped**: it
overlaps R4 conceptually and is noisier per-bar on M1. See Sec. 6.3.)

---

## 4. Evaluation Metrics

Metrics are **pooled period-level** (DeepScalper Sec. 5.2): per-bar returns of every
session in a phase are concatenated into one sequence; one eval pass = one row.
Evaluation uses the **same metric set for every reward variant** (the reward
only affects training, never how results are scored).

Notation: $r_i$ = per-bar log return; $d_k$ = per-day return (sum of that day's
$r_i$); the phase has $D$ days. Equity curve $E_i = \exp\!\big(\sum_{j \le i} r_j\big)$.

| Metric | Definition |
|---|---|
| `total_return` | $\exp\!\big(\sum_i r_i\big) - 1$ over the phase |
| `sharpe` | $\dfrac{\operatorname{mean}(d_k)}{\operatorname{std}(d_k)} \times \sqrt{252}$ (daily-aggregated, annualized) |
| `sortino` | $\dfrac{\operatorname{mean}(d_k)}{\operatorname{std}\big(d_k \mid d_k < 0\big)} \times \sqrt{252}$ — denominator is **std** of down days |
| `mdd` | $\min_i \dfrac{E_i - \max_{j \le i} E_j}{\max_{j \le i} E_j}$, continuous within the phase |
| `trades` | count of contiguous non-zero position blocks |
| `winrate` | wins / total trades (pooled) |
| `avg_trade_pnl` | mean completed-trade log return |
| `turnover` | $\sum_t \lvert a_{t} - a_{t-1} \rvert$ (sum of absolute position changes) |
| `avg_holding_time` | mean bars held per completed trade |
| **`final_equity`** | mean end-of-episode equity in dollars — *new, portfolio metric* |
| **`max_dd_dollar`** | worst peak-to-trough equity drop, in dollars and % — *new* |
| **`ruin_rate`** | fraction of episodes that hit $\text{equity} \le 0$ — *new* |

Best-checkpoint selection uses a configurable val metric (default `sortino`).

---

## 5. Experiments

| Exp | Question | Design |
|---|---|---|
| **0 — Reward Selection** *(new)* | Which reward family (R1/R2/R4) trains the best agent under equal conditions? | 3 rewards × 3 algos (DDQN/A2C/PPO) × 3 seeds = **27 runs**. **No HPO** — fixed hyperparameters (DDQN: prior HPO best; A2C/PPO: literature defaults). Reward normalization on. Winner = best **mean rank across the 3 algos** on validation (robust to algo idiosyncrasy). |
| **1 — Algorithm Comparison** | Which DRL algorithm learns best? | DDQN vs A2C vs PPO, **using the Exp-0 winning reward**, with full inner-CV HPO per algorithm, 3 seeds, test once per algo. |
| **2 — Action Space** | Does a finer action set help? | `[-1,0,+1]` vs `[-1,-0.5,0,+0.5,+1]` vs 9-action, on the Exp-1 winning algo + winning reward. |
| **3 — Sequential Models** | Do LSTM/GRU beat MLP? | MLP vs LSTM vs GRU on the Exp-2 winner. Causal sequence windows only. |
| **Final Baselines** | Better than naive/rule-based after cost? | Flat / Long / Short / Random / MA-crossover on test, same env/cost/EOD rule. |

**Why Exp 0 has no HPO:** HPO is the most expensive step (~1.5 h/algo).
Exp 0 only needs to rank reward *families* under equal conditions; the winner is
then fully tuned in Exp 1. Honest limitation: Exp 0 answers "which reward family
is more promising at fixed, normalized conditions", not "which reward is best
when fully tuned" — that is Exp 1's job.

---

## 5.1 Exp 0 Results (run 2026-05-16, **v2** after spec corrections)

> **Provenance.** 27 runs (3 rewards × 3 algos × 3 seeds × 500k timesteps).
> Wall-clock 594 minutes on a single GPU. Configuration deviations from the
> original Sec. 6.4 spec are documented inline and in JOURNAL.md (entry of
> 2026-05-16) — most notably **reward normalization was disabled** after an
> A/B test disproved it, and a **per-reward learning-rate regime** replaced
> the single shared lr. See "Lessons" below.

### Winner: **R4** (mean rank 1.67 across DDQN/A2C/PPO)

![Exp 0 ranking — mean val_total_return per (reward, algo)](../runs/exp0/plots/exp0_ranking.png)

| Reward | DDQN rank | A2C rank | PPO rank | **Mean rank** | Conclusion |
|---|---|---|---|---|---|
| **R4** (P&L − DD penalty) | **#1** | #2 | #2 | **1.67** | 🥇 Winner |
| R1 (log-return) | #2 | #1 | #3 | 2.00 | second |
| R2 (raw $ P&L) | #3 | #3 | #1* | 2.33 | third |

\* R2's PPO rank=#1 is a [rank-from-below pathology](https://en.wikipedia.org/wiki/Survivorship_bias):
all 3 R2+PPO seeds collapsed to do-nothing (test return = 0.0%), and "do nothing"
ties at rank #1 against alternatives that also yielded near-zero values. The
`min_trades ≥ 50` filter used in Exp 1's HPO would disqualify these trials; Exp 0
has no such filter and thus reports the raw mean rank for transparency.

### Test-set mean return (75 days, 2026-01-15 → 2026-04-30)

![Reward × algo heatmap — mean test total_return](../runs/exp0/plots/reward_x_algo_heatmap.png)

The heatmap makes the **R4 + DDQN cell the only positive-and-non-trivial result**.
R1 + PPO at −46.97% is the catastrophic over-trading mode (4,917-9,707 trades on
test, eating ~$80-100 of spread cost cumulatively).

### Top 4 runs by Sharpe

![Equity curves of top 4 runs on test set](../runs/exp0/plots/equity_curves_top.png)

| Rank | Run | Final eq | Return | Sharpe | Sortino | trades | winrate | mdd |
|---|---|---|---|---|---|---|---|---|
| 1 | **r2_ddqn_s1337** | $10,012.59 | +9.28% | **+1.53** | **+2.56** | 653 | 0.479 | $461 (4.6%) |
| 2 | **r4_ddqn_s1337** | $10,009.48 | +6.80% | **+1.18** | **+2.20** | 433 | 0.471 | $430 (4.3%) |
| 3 | **r4_ddqn_s42** | $10,008.48 | +5.97% | **+1.01** | **+1.57** | 665 | 0.469 | $584 (5.8%) |
| 4 | r1_ddqn_s1337 | $10,001.14 | +0.81% | +0.51 | +0.20 | 84 | **0.714** | $356 (3.6%) |

All four top runs are **DDQN**. r4_ddqn produces a more consistent Sharpe across
seeds (+1.01 to +1.18 on 2/3 seeds) than r2_ddqn (a single +1.53 outlier),
which is what shifts the mean-rank win to R4 despite R2 having the single
best individual run.

### Collapse audit — 12/27 runs went flat or near-flat

![Collapse audit — test trades per (reward, algo, seed)](../runs/exp0/plots/collapse_audit.png)

A2C and (especially) PPO struggled badly with this MDP at the per-reward lrs
that worked in mini-test:

| Algo | R1 collapses | R2 collapses | R4 collapses | Pattern |
|---|---|---|---|---|
| DDQN | 0 | 0 | 0 | none — bias = "ε-greedy random exploration finds non-zero trades" |
| A2C  | 1 (seed 42) | 1 (seed 1337) | 1 (seed 1337) | seed-specific — policy gradient signal at default A2C lr (0.0007 for R1, 1e-5 for R2/R4) is borderline; ~1/3 seeds land in the flat attractor |
| PPO  | 0 (but **over-trades** 5k-10k) | 3 | 3 | structural — PPO at our chosen lr (0.0003 for R1, 1e-6 for R2/R4) either explodes into over-trading (R1) or implodes into flat (R2/R4) |

PPO+R1 produced the **opposite** failure mode: 4,917-9,707 trades per 75-day
test (≈65-130 trades per day) at winrate 0.34-0.59 — the signed reward signal
from R1's tiny per-bar log-return drives K-epoch SGD into a hyperactive policy
that bleeds spread cost.

### Per-run test return (full grid)

![Per-run test return — 27 runs](../runs/exp0/plots/test_return_by_run.png)

### Lessons & deviations from the original Sec 6.4 spec

The original Sec 6.4 mandated `running-std reward normalization` on every
variant. **A pre-Exp-0 A/B test (DDQN R1 seed=42, 500k timesteps) disproved
this:** with normalization on, 0/16 val evals had any trades — instant collapse
to flat. Mechanism: R1's per-bar log-return is ~1e-4; the normalizer std starts
tiny (n<10 samples) and amplifies R1 by ~10^4-10^5 in the first few thousand
steps. The Q-network sees gigantic synthetic rewards for the action it last
sampled (~uniform at init), the "flat" action has Q=0/std=0 as a stable target,
and the policy collapses. Disabling normalization recovered 16/16 evals with
trades > 0 and best val_total_return = +0.41%.

The replacement — `lr_per_reward` overrides in each algo's config block —
calibrates the learning rate to the reward magnitude:

| algo | R1 lr | R2 lr | R4 lr |
|---|---|---|---|
| DDQN | 0.0045 | 1e-5 | 1e-5 |
| A2C | 0.0007 (default) | 1e-5 | 1e-5 |
| PPO | 0.0003 (default) | 1e-6 | 1e-6 |

These three lr A/B/C tests (one per algo × R1+R2) are recorded in `JOURNAL.md`
2026-05-16. Sec 6.4 of this document is preserved as written for historical
record; for current and future experiments, **normalization is off and per-reward
lr is on**.

### Caveat for the algorithm comparison

Exp 0's headline finding — R4+DDQN beats everything — is **partly an artifact
of A2C and PPO being un-tuned at the per-reward lr boundary we picked from
100k mini-tests**. PPO+dollar-reward collapses 6/6 seeds; A2C collapses 3/9
seeds. The Exp 1 algorithm comparison must address this before declaring
DDQN "the best algorithm": HPO over a wider lr range, entropy schedule for
A2C/PPO, dueling Q-net for DDQN, and (most importantly) γ tuning are all
plausible fixes — see Sec. 8 for the formal anti-collapse plan that precedes
Exp 1.

---

## 6. Reward Function Details

Reward families were chosen from a survey of the project paper library
(`research_papers/`). Selection criteria: must be a **dense per-step signal**
(M1 credit assignment; sparse episodic Sharpe/MDD/CVaR rejected as primary
signals — TDQN, DRQN, DeepTrader all note this) and compatible with a
fixed-lot dollar-P&L env.

### 6.1 R1 — Log-return of equity (baseline / control)

`r_t = log(equity_{t+1} / equity_t)` after spread cost.
- **Origin:** Goluža et al. 2406.08013 (the prior project reward), Jiang et al.
  1706.10059, Huang DRQN 1807.02787.
- **Role:** the control. Additive across steps (Σ log = log compound), scale-free.
  Every other reward must beat this to justify itself.
- **Trade-off:** compresses large gains; no risk awareness; tiny per-bar
  magnitude (~±1e-4).

### 6.2 R2 — Raw net dollar P&L (matches project intent)

`r_t = equity_{t+1} − equity_t` (dollars, after spread cost).
- **Origin:** DeepScalper 2201.09058, FineFT 2512.23773, MaxAI
  ssrn_5761402, Cao et al. 2103.16409 (accounting P&L).
- **Role:** the literal "money in the portfolio" objective the project intends.
- **Trade-off:** large, unbounded scale → noisier gradients; not additive like
  log; no risk awareness.

### 6.3 R4 — P&L minus drawdown penalty (risk-aware, fits the env)

`r_t = Δ$_t − β · max(0, DD_t − dd_thresh)`, with `equity ≤ 0` termination.
- **Origin:** CMDP execution Borjigin & He 2510.04952, regime-aware Raj
  2509.14385, DeepTrader 2021 (−MDD reward gave best MDD in their ablation).
- **Role:** capital-preservation aware; aligns with the env's ruin termination
  (a CMDP-style hard constraint already exists, R4 adds a soft drawdown penalty).
- **Trade-off:** drawdown is path-dependent → delayed credit assignment; adds
  `β`, `dd_thresh` hyperparameters (tuned only in Exp 1 for the winner).
- **Dropped alternative — R3 (mean-SD / vol-penalized):** conceptually overlaps
  R4 ("P&L minus a risk term") but uses a running std that is noisy on M1 and
  needs a window hyperparameter. R4 is kept instead because it ties naturally to
  the env's existing ruin constraint.

### 6.4 Reward normalization (training only)

R1 (~±1e-4/bar) and R2/R4 (~±$1–30/bar) differ in scale by ~10⁴–10⁵×. A single
learning rate cannot be fair to all three: an `lr` tuned for R1 makes R2/R4
diverge (NaN), and vice-versa. To compare reward *shape* rather than reward
*scale*, every reward is divided by a **running standard deviation of the
reward stream** before being fed to the optimizer (same principle as the
existing `normalize_advantage` in A2C/PPO).

- The normalization is applied to the **whole reward including the R4 penalty**
  (running std over the final `Δ$ − penalty`). This keeps the gradient scale of
  all three rewards equal — the top priority for a fair Exp 0.
- Consequence (stated honestly): R4's drawdown penalty is rescaled, so its
  punitive effect is softened in Exp 0. Exp 0 therefore asks "is a
  drawdown-aware reward family more promising than plain P&L", not "is a fully
  calibrated R4 best". Calibrating `β`/`dd_thresh` is Exp 1's job (HPO on the
  winning reward).
- **Evaluation is never normalized** — all variants are scored with the Sec. 4
  metric set on raw equity.

---

## 7. Controlled Settings & Validity

Fixed across all conditions unless it is the explicit variable under study:
dataset, 750-day window, 600/75/75 split, episode = 1 day, execution model,
spread-cost model, EOD flatten, ruin rule, feature set, deterministic eval
policy, best-checkpoint = validation metric, test reported only once after
model selection.

Leakage / validity checklist:
1. Chronological split; never shuffle the split order.
2. No normalization statistics fit on val/test.
3. No hyperparameter or checkpoint selection on test.
4. Never shuffle timesteps within a day.
5. Rolling indicators use only past/current data.
6. LSTM/GRU windows contain no future bars.
7. Spread cost applied to all DRL methods and all baselines.
8. Same seeds `[42, 1337, 2026]` across comparable conditions.
9. Report mean ± std across seeds.
10. Comparable training budget across algorithms within an experiment.

---

## 8. Anti-Collapse Plan (precedes Exp 1, 2026-05-16)

Exp 0 v2 revealed that 12/27 runs collapsed to do-nothing flat policies. A
literature review by two parallel sub-agents over the project paper library
identified four high-confidence techniques and their evidence:

| # | Technique | Cite | Code change | Expected fix |
|---|---|---|---|---|
| 1 | **Lower discount factor γ** from 0.99 → ~0.3 | Zhang/Zohren/Roberts 2019 (`1911.10107`, Oxford-Man, Table 1) use γ=0.3 for DQN/PG/A2C on futures. | 1 line in `config.yaml`. Sweep γ ∈ {0.1, 0.3, 0.7, 0.9}. | At γ=0.99 the value of "flat" — a zero-reward sequence — is a stable low-variance target that dominates the noisy Q(trade) target. Lowering γ raises target SNR. |
| 2 | **Hindsight reward bonus** | DeepScalper §4.2 (Sun et al. 2022, `2201.09058`). Table 4 ablation: TR 3.5% → 6.97% on M1 futures. | `r_t += w · (close[t+h] − close[t]) · pos_t` during training only (no leakage at deploy). ~20 LOC in `src/env.py`. | Adds a dense look-ahead signal that turns trading actions into positive expected reward even before the policy has discovered profitable patterns. |
| 3 | **Dueling Q-network** | Wang et al. 2016; DeepScalper §4.1 and Zhang/Zohren/Roberts §3.2 use it as default. | `src/ddqn.py::QNet` — split final layer into V(s) + A(s,a) − mean A. ~15 LOC. DDQN-only. | V(s) updates on every transition regardless of action; V(trending day) is learnable even when the agent hasn't yet traded. Breaks the symmetric Q≈0 collapse. |
| 4 | **Entropy schedule + policy-head init** | Engstrom 2020 (`2005.12729`) "Implementation Matters"; Andrychowicz `2006.05990` "What Matters in On-Policy RL". | A2C/PPO `entropy_coef` linearly annealed 0.05 → 0.005. Audit orthogonal-init gain 0.01 on policy head. ~30 LOC. | PPO collapse to flat is a near-deterministic-policy degenerate distribution; a higher entropy penalty makes it actively costly. |

**Skipped** by sub-agent consensus (do not pursue): ICM/RND curiosity (volatility
novelty correlates with losses, not profits), Noisy Networks (no financial-RL
paper uses it), prioritized replay alone (amplifies negative signal), action-balanced
forced exploration (introduces deterministic losing trades).

A broader catalogue of DRL extensions surveyed for this project — DDQN
variants (PER, Dueling, n-step, distributional, Munchausen, NoisyNet), A2C
variants (recurrent A2C, action augmentation, DeepScalper auxiliary heads,
SAC-Discrete), PPO variants (VC-PPO, return-based reward scaling, KL
early-stop, PPG, GRPO, recurrent PPO), and alternative algorithm families
beyond the baseline trio (SAC-Discrete, QR-DQN, Decision Transformer, CQL,
risk-sensitive RL, model-based, hierarchical, evolutionary, imitation
hybrids) — is in **`proposal/algorithm_extensions.md`**. Each entry cites
its origin paper. Future Exp 0.5 / Exp 1.5 / Exp 3 designs draw from that
catalogue.

**Order of attack:**

1. **Exp 0.5a — γ sweep alone** (~3h, 9 runs at 500k). Drop-in. If R4+DDQN
   keeps its lead and PPO escapes the flat attractor, γ was the bottleneck.
2. **Exp 0.5b — hindsight bonus on top of best γ** (~3h, 9 runs). Adds 20 LOC.
3. **Exp 0.5c — dueling Q-net + entropy schedule** (~3h, 9 runs).

Each Exp 0.5 round uses the same 3 rewards × 3 algos × 3 seeds = 9 runs grid;
the winner of each round defines the baseline for the next. If round 1 already
solves the collapse (≥6/9 runs profitable, ≤2/9 flat), rounds 2-3 can be folded
into Exp 1's HPO instead of run separately. The full anti-collapse program is
estimated at 9-15 h wall-clock, similar to one Exp 0 v2.

**Then Exp 1** (algorithm comparison + per-algo HPO) runs on the anti-collapse-fixed
config, so the DDQN-vs-A2C-vs-PPO question is fair.

### 8.1 Exp 0.5a Results — γ sweep on DDQN (2026-05-16)

12 DDQN+R4 runs (4 γ × 3 seeds × 300k steps; reduced from the proposed 500k
to match Exp 0 v2 speed budget after observing collapse is detectable by
~150k steps). Wall time: **63.4 min** with N=4 parallel subprocess workers.

| γ | Collapse (val_trades < 50) | Mean test_ret | StdDev | Mean Sortino | Mean trades |
|---|---|---|---|---|---|
| **0.30** | **0/3** | **+0.0223** | 0.041 | **+1.03** | 612 |
| 0.50 | 1/3 | +0.0019 | 0.013 | -0.08 | 134 |
| 0.90 | 0/3 | **-0.0619** | **0.101** | -1.26 | 713 |
| 0.99 | 0/3 | -0.0193 | 0.038 | -0.21 | 517 |

**Winner: γ=0.30 across all three dimensions** — best mean return, best mean
Sortino, lowest collapse rate. The result reproduces Zhang/Zohren/Roberts 2019
qualitatively: γ=0.3 is the only setting that yields a positive risk-adjusted
return on average, and the only one that escapes the do-nothing attractor on
every seed. Best individual run `g0.3_s2026` returned **+7.90% with Sortino
+3.57** on the held-out test split — by a wide margin our first clearly
profitable agent on this MDP.

The γ=0.9 result is the most informative *negative*: highest variance
(StdDev 0.101), one seed lost 20.2% with 1521 trades. γ=0.9 sits in an
unstable zone where the agent trades enough to lose money but not enough
to learn — worse than the conservative γ=0.99 baseline.

Action: `config.yaml dqn.gamma: 0.99 → 0.30`. Plots and per-run details in
`runs/exp05a/`; full discussion in JOURNAL.md 2026-05-16 entry.

**Next:** Exp 0.5b is now contingent on Phase 1b (A2C/PPO at γ=0.3). If the
γ effect generalizes across algorithms, the anti-collapse program may
collapse from 3 rounds into 1, and Exp 1 (HPO) can start immediately on a
clean baseline. If A2C/PPO still collapse at γ=0.3, the program proceeds as
planned to hindsight (0.5b) and dueling/entropy (0.5c).

---

## 9. Out of Scope / Future Work

- **Capital curriculum** (e.g. train at $10k, then anneal toward a low-capital
  high-risk regime such as $100) — explicitly a future experiment, not the main
  study; needs A/B vs the $10k baseline.
- **Dynamic / risk-based lot sizing** (Kelly, fixed-fractional, vol-scaled) —
  current design is fixed 0.01 lot only.
- **Broker-realistic margin/leverage** (margin level %, stop-out, swap/financing)
  — current design uses a single `equity ≤ 0` ruin threshold only.
- **Volatility-scaled reward (R3-family / Zhang-Zohren)** — conflicts with the
  fixed-lot mandate; revisit only if variable sizing is added.
- **iRDPG-style warm-start from MA-crossover** (Liu et al. AAAI 2020,
  `5587_13_8812`) — viable next-line fix if Exp 0.5 rounds 1-3 don't solve PPO
  collapse; behavior-cloning loss for 1k episodes from `MACrossoverBaseline`.
- **Dormant-neuron monitoring + ReDo** (Sokar et al. ICML 2023) — diagnostic
  telemetry only; full recycling only if dormancy > 20% per layer.

---

## 10. Historical Note

- The project was first presented with a **return-based MDP** (reward = per-bar
  log-return after a relative commission; no capital, no lots, no ruin). That
  spec lives in `DRL_proposal_6509_6571_OLD_presented.pdf` and earlier
  JOURNAL.md entries.
- On switching to this portfolio-based MDP, the following became **invalid** and
  must be rebuilt/rerun: `src/env.py` (capital/lot/ruin/reward modes/
  normalization), state dim 15 → 16, agent input layers, `src/data.py`
  (spread → real cost), `config.yaml` (capital/lot/reward sections), and **all
  prior `runs/` and HPO outputs** (return-based, not comparable).
- Reward-function survey of `research_papers/` produced the R1/R2/R4 shortlist
  in Sec. 6.
