# Algorithm Extensions — Research Notes

> **Purpose.** Catalogue of DRL techniques worth considering on top of the three
> Exp 0 baselines (DDQN / A2C / PPO), compiled from four parallel sub-agent
> reviews of the project paper library (`research_papers/`) plus broader RL
> literature. Generated 2026-05-16 after Exp 0 v2 completion.
>
> **How to read this.** Each idea names the paper that proposed it. Tier 1
> ("Top picks") = strongest evidence for our M1 intraday + portfolio MDP
> setup. Tier 2 = solid options worth A/B'ing. Tier 3 = mentioned for
> completeness. Rejected items state *why* they were rejected so we don't
> re-evaluate them later.
>
> **Scope reminder.** Per user request 2026-05-16, the algorithm choice is
> **not** locked to DDQN/A2C/PPO. Section 5 (Beyond the baselines) covers
> entirely different algorithm families.
>
> **What's already in our paper library** is marked with `[library]` after
> the citation.

---

## Cross-cutting top picks (across all algorithms)

These three techniques surface in multiple sub-agent reports and have the
strongest published evidence for our specific failure modes (collapse to flat,
short-termism, over-trading). They are described in detail in the per-algorithm
sections below.

| # | Technique | Origin paper | Fixes | Applies to |
|---|---|---|---|---|
| 1 | **DeepScalper hindsight bonus** | Sun et al. 2022, arXiv:2201.09058 `[library]` | Short-termism, collapse to flat (Table 4 ablation: TR 3.5%→6.97% on M1 futures) | All 3 algos |
| 2 | **Lower γ (e.g. 0.3 instead of 0.99)** | Zhang, Zohren & Roberts 2019, arXiv:1911.10107 `[library]` | Q(flat) being a low-variance attractor under long horizons | All 3 algos (config change only) |
| 3 | **Action augmentation (counterfactual TD/PG update)** | Huang 2018, arXiv:1807.02787 `[library]`; Asadi et al. 2017 "Mean Actor-Critic" arXiv:1709.00503 | Sample efficiency, exploration collapse — exploits *exogenous-price* property of our env | All 3 algos (env helper needed) |

The full anti-collapse plan (γ sweep → hindsight → dueling → entropy schedule
→ behaviour-clone warm-start) is in **`PROPOSAL.md §8`**.

---

## 1. DDQN Extensions

Current baseline: vanilla Double DQN, MLP `16 → [128,128] → 3`, uniform replay,
hard target update, ε-greedy linear decay, Adam ε=1.5e-4, Huber loss.

### Tier 1 — High evidence for M1 trading

1. **Prioritized Experience Replay (PER)** — Schaul et al. 2016, arXiv:1511.05952 (ICLR 2016). Sample by `|TD error|` rather than uniform. Used inside **DeepScalper** (arXiv:2201.09058 `[library]`) and the SDAEs-LSTM DDQN of Li et al. 2019. Most M1 bars are flat; PER concentrates learning on rare high-information regime-shift transitions. **~150 LOC, 4–6 h.** Replace `ReplayBuffer` with a SumTree, add IS weights to loss, write back `|td|+ε`.

2. **Dueling Network Architecture** — Wang et al. 2016, arXiv:1511.06581 (ICML 2016). Decompose `Q(s,a) = V(s) + (A(s,a) − mean_a A)`. With only 3 actions, V absorbs most variance and advantage learning becomes cleaner. DeepScalper uses a Branching Dueling variant (Tavakoli et al. 2018, AAAI). **~30 LOC, 1–2 h.** Two heads replace `fc_out`. Composes trivially with PER + Double DQN.

3. **n-step returns** — Sutton 1988; integrated in Rainbow, Hessel et al. 2018, arXiv:1710.02298 (AAAI 2018). Bootstrap from `r_t + γr_{t+1} + … + γ^(n-1)r_{t+n-1} + γ^n max Q_target(s_{t+n}, ·)`. **Li et al. 2019** (IEEE Access, DOI 10.1109/ACCESS.2019.2932789) use **n=10 explicitly on 1-min financial data** with large gains over 1-step. **~80 LOC, 3 h.** Deque of recent transitions; recompute target with `γ^n`.

### Tier 2 — Architecture / loss variants

- **C51 (Distributional DQN)** — Bellemare, Dabney & Munos 2017, arXiv:1707.06887 (ICML 2017). Categorical distribution over 51 return atoms. Fixed `(V_min, V_max)` support is awkward when R2 dollar reward scale varies day-to-day; prefer QR-DQN below.
- **QR-DQN** — Dabney et al. 2018, arXiv:1710.10044 (AAAI 2018). Quantile regression; no support tuning. Cleaner fit for R2's unknown scale. **~200 LOC, 6–8 h.** Quantile Huber loss replaces standard Huber.
- **IQN (Implicit Quantile Networks)** — Dabney et al. 2018, arXiv:1806.06923 (ICML 2018). Samples τ∼U(0,1) at every forward pass; predicts that quantile. Best-in-class for risk-sensitive DQN — **CVaR objective expressible directly** by selecting actions on lower-quantile expectations. **~250 LOC, 10 h.**
- **Munchausen DQN (M-DQN)** — Vieillard et al. 2020, arXiv:2007.14430 (NeurIPS 2020). Add scaled log-policy bonus `α·τ·log π(a|s)` to the immediate reward (π = softmax of current Q). Implicit KL regularization; often beats Rainbow with one extra line of code. **~20 LOC, 2 h.** Addresses Q-overestimation and policy churn — both visible in our DDQN's noisy `mean_q` curve.
- **NoisyNets** — Fortunato et al. 2018, arXiv:1706.10295 (ICLR 2018). Replace ε-greedy with learned per-layer Gaussian weight noise. State-conditional exploration; eliminates ε-decay hyperparameter. **~80 LOC, 3 h.**

### Tier 3 — Replay / exploration variants

- **DQfD (Deep Q-from-Demonstrations)** — Hester et al. 2018, arXiv:1704.03732 (AAAI 2018). Pre-fill buffer with expert trajectories (e.g. MA-crossover or oracle hindsight). Adds supervised margin loss. Trading instance: Fang et al. 2021 "Universal Trading", arXiv:2103.10860 `[library]`. **Risk:** our baselines are weak (MA-crossover loses on XAUUSD M1), so demo-bias may *hurt*.
- **Bootstrapped DQN** — Osband et al. 2016, arXiv:1602.04621 (NeurIPS 2016). K parallel Q-heads, each trained on a bootstrap mask; Thompson-sampling-like exploration. Probably overkill at 3 actions.
- **Boltzmann/softmax exploration** — temperature-scaled `softmax(Q/τ)`. **~5 LOC, 1 h.** Cheap A/B vs ε-greedy.

### Trading-specific DDQN tricks

- **Hindsight bonus reward** — DeepScalper §4.2, Sun et al. 2022, arXiv:2201.09058 `[library]`. `r_t += w · (close[t+h] − close[t]) · pos_t`, training only. Best `w≈0.1`, `h≈120` bars in their ablation. **~10 LOC in env, 1 h.** Compatible with R1/R2/R4.
- **Volatility-prediction auxiliary head** — DeepScalper §4.4. Small MLP predicts realized vol over next h bars; multitask loss `L_q + η·L_vol`. Their ablation: **+0.38 SR** on stock index. **~50 LOC, 3 h.**
- **SDAE state denoising** — Li et al. 2019 (IEEE Access). Pre-train a Stacked Denoising Autoencoder on OHLCV+indicators; freeze, then DDQN trains on the embedding. **~150 LOC + pretraining script, 8 h.** Best paired with Exp 3 LSTM.
- **Position-extended action space** — Li et al. 2019. Action set `{−n,…,n}` as shares-held; matches PROPOSAL Exp 2 (5/9 actions).
- **Data augmentation (signal shift, low-pass, noise)** — Théate & Ernst 2021 (TDQN), arXiv:2004.06627. Helps when train data is small (600 days is borderline). **~80 LOC, 3 h.**

### Rejected for DDQN

- **HER** — Andrychowicz et al. 2017, arXiv:1707.01495. Goal-conditioned MDP only; our reward is dense P&L without goals.
- **Retrace(λ)** — Munos et al. 2016, arXiv:1606.02647. Overlaps n-step+PER with more complexity; revisit only for stale-data replay.
- **ICM / RND intrinsic motivation** — Pathak 2017, arXiv:1705.05363; Burda et al. 2018, arXiv:1810.12894. Novelty is anti-signal in trading (unusual market = high vol = high spread cost).
- **Full Rainbow** — Hessel et al. 2018, arXiv:1710.02298. Too many simultaneous changes for our ablation discipline; combine 2-3 components at a time instead.

---

## 2. A2C Extensions

Current baseline: shared trunk MLP `16 → [128,128]` Tanh + policy/value heads,
GAE(λ=0.95), per-rollout advantage normalization, episode-aligned rollout
(1 day), one Adam step/rollout, entropy_coef=0.01, `max_grad_norm=0.5`.

### Tier 1 — High evidence for M1 trading

1. **Recurrent A2C (LSTM/GRU trunk)** — Ponomarev, Oseledets & Cichocki 2019, "Using Reinforcement Learning in the Algorithmic Trading Problem," J. Comm. Tech. Electronics 64(12). Wrap features with `nn.LSTM` between input and the two heads; reset `(h,c)` at episode boundary. Their ablation on MOEX:RTSI 1-min futures shows LSTM variant beats LSTM-less by a wide margin and reports **66% p.a. net of commission**. Addresses partial observability our 16-dim Markov state misses (volatility regime, intraday phase). **~80 LOC, 4–6 h.** Reuses GAE and rollout loop unchanged.

2. **DeepScalper auxiliary heads (vol prediction + hindsight bonus)** — Sun et al. 2022, arXiv:2201.09058 `[library]`. `ActorCritic.forward` returns `(logits, value, vol_pred)`; loss adds `λ_aux · MSE(vol_pred, vol_target)`. Hindsight bonus in env reward. Addresses A2C's "selective-but-unprofitable" pattern on R2/R4. **~120 LOC, 6–8 h.**

3. **Action augmentation (counterfactual PG)** — Huang 2018, arXiv:1807.02787 `[library]`; Asadi et al. 2017 "Mean Actor-Critic," arXiv:1709.00503. Since price is exogenous to action, compute reward for *all 3 actions* at every bar; replace single-sample PG with `−Σ_a π(a|s)·A(s,a)`. Likely the **highest-leverage single trick** for our setup; the agent gets a learning signal even on actions it didn't take. **~60 LOC, 3–4 h.** Trading-specific.

### Tier 2 — Algorithm variants

- **A3C (asynchronous)** — Mnih et al. 2016, arXiv:1602.01783 (ICML 2016). Multiple workers, async gradient push. Wall-clock benefit only; not a sample-efficiency or stability fix. Useful if compute is the bottleneck.
- **ACER (off-policy A2C with Retrace)** — Wang et al. 2017, arXiv:1611.01224 (ICLR 2017). Replay buffer + truncated importance sampling. **~400 LOC, 3–5 days.** High instability risk on noisy reward.
- **IMPALA / V-trace** — Espeholt et al. 2018, arXiv:1802.01561 (ICML 2018). Clipped IS off-policy correction; more stable than ACER. **~200 LOC, 1–2 days.**
- **SAC-Discrete** — Christodoulou 2019, arXiv:1910.07207. Discrete-action SAC with auto-tuned entropy `α`. **Directly fixes the flat-collapse via entropy floor.** Off-policy → sample-efficient. **~250 LOC, 2 days.** *Strong recommendation.*

### Tier 2 — Variance reduction & critic improvements

- **Distributional critic (C51 value head)** — Bellemare et al. 2017, arXiv:1707.06887. Captures fat-tailed P&L (a known R2 issue: one ruinous day dominates MSE). **~150 LOC, 1 day.**
- **GAE λ sweep** — Schulman et al. 2016, arXiv:1506.02438 (ICLR 2016). Our λ=0.95 is cargo-cult default; on 1379-bar episodes with noisy per-bar rewards, λ ∈ [0.85, 0.97] should be re-swept. **5 LOC, 6 h HPO.**
- **PopArt value/reward normalization** — van Hasselt et al. 2016, arXiv:1602.07714 (NeurIPS 2016). Normalizes value output while preserving unnormalized policy gradient — more principled than our env-level reward wrapper for R2 (spans 4 orders of magnitude). **~80 LOC, 1 day.**

### Tier 3 — Exploration & entropy

- **Linear/cosine entropy-coef decay** — Mnih 2016 A3C used constant 0.01; modern practice anneals. Engstrom et al. 2020 "Implementation Matters," arXiv:2005.12729 `[library]` shows it's one of the highest-impact code-level choices. **~10 LOC, instant HPO.**
- **Adaptive entropy via KL target** — SAC's auto-α (Haarnoja et al. 2018, arXiv:1812.05905). Maintain α so mean policy entropy ≈ target. **~30 LOC, 4 h.**

### Trading-specific actor-critic tricks

- **Position-controlled action space** — Li, Zheng & Zheng 2019 IEEE Access (DOI as above). A3C+LSTM with graded position deltas. Relevant for PROPOSAL Exp 2.
- **Adversarial critic** — Liang et al. 2018 "Adversarial DRL in Portfolio Management," arXiv:1808.09940 `[library]`. Train V(s) against an adversary that perturbs price; improves robustness to regime shift. **~150 LOC, 1–2 days.**
- **DeepTrader asymmetric risk head** — Wang et al. 2021 AAAI `[library]`. Splits critic into return-prediction + risk-prediction streams. Overlaps with DeepScalper auxiliary task.

### Rejected for A2C

- **ACKTR / K-FAC** — Wu et al. 2017, arXiv:1708.05144. 2nd-order optimization buys little on a 16-dim state.
- **R2D2-style recurrent replay** — Kapturowski et al. 2019, ICLR `[library]`. Designed for Q-learning at scale; overkill for on-policy day-long rollouts.
- **Full PPO-style clip inside A2C** — that's literally PPO. Don't blur the line.
- **RND / curiosity** — Burda et al. 2018, arXiv:1810.12894. Same reason as DDQN section.

---

## 3. PPO Extensions

Current baseline: same trunk as A2C, GAE(λ=0.95), per-minibatch adv-norm,
clipped surrogate clip_range=0.2, K=10 SGD epochs, minibatch_size=128,
`max_grad_norm=0.5`, entropy_coef=0.01. **All 3 R2 seeds and 3 R4 seeds
collapsed to flat in Exp 0 v2; R1 over-trades 5-10k trades on test.**

### Tier 1 — Direct fixes for our collapse/over-trade

1. **Return-based reward scaling** — Engstrom et al. 2020 "Implementation Matters," arXiv:2005.12729 `[library]`; Huang et al. 2022 "The 37 Implementation Details of PPO," ICLR Blog (detail #28). The OpenAI Baselines wrapper divides reward by a running std of the *discounted return sum*, not raw reward. Engstrom shows this is the single biggest swing in PPO performance. We currently wrap at the env level; PPO needs the return-based variant **inside the rollout buffer**. Fixes R2/R4 PG variance. **~30 LOC, 2 h.**

2. **VC-PPO: decoupled GAE λ + value pre-training** — Yuan et al. 2025 "What's Behind PPO's Collapse in Long-CoT? Value Optimization Holds the Secret," arXiv:2503.01491. Diagnoses **PPO collapse** as value-function failure: value bias → corrupted advantages → policy degenerates. Fix: larger λ for value (e.g. 0.99), smaller for policy (e.g. 0.9); pre-train value before policy. Our 1379-bar episodes with sparse meaningful returns are structurally similar to long-CoT. **~40 LOC + 20 LOC for value pre-train, 4 h.** *Most theoretically motivated fix for our collapse.*

3. **KL early-stop + clip-range schedule** — Schulman et al. 2017, arXiv:1707.06347 (KL-penalty variant); CleanRL/ICLR-blog practice. We have `kl_early_stop` plumbed but **disabled**. Enable `target_kl≈0.015` to stop the K=10 epochs early when policy moves too far per rollout — exactly the R1 over-trade regime. Pair with linear clip-range anneal 0.2→0.05. **~10 LOC, 1 h.**

### Tier 2 — Algorithm variants

- **PPG (Phasic Policy Gradient)** — Cobbe et al. 2021, arXiv:2009.04416 (ICML 2021). Alternate policy and value optimization phases; value gets many epochs with auxiliary distillation while policy stays on-policy. Directly addresses VC-PPO's diagnosed failure mode. **~150 LOC, 1 day.**
- **GRPO (Group Relative Policy Optimization)** — Shao et al. 2024 "DeepSeekMath," arXiv:2402.03300. Drops value network entirely; advantage = `(reward − mean of K sampled-trajectory rewards) / std`. Awkward fit: GRPO assumes K trajectories from same state — for trading days, this means re-running the same date K times. Doable with deterministic env. **~80 LOC, 4 h.**
- **KL-penalty PPO (PPO-penalty)** — Schulman et al. 2017 (the "other" PPO variant). Lagrangian KL penalty with adaptive β. Sometimes preferred when reward magnitudes vary. **~20 LOC, 1 h.**
- **SPO (Simple Policy Optimization)** — Xie et al. 2025, arXiv:2401.16025 (ICML 2025). Modified clip that better constrains the probability ratio inside the trust region. **~30 LOC, 2 h.**
- **TRPO** — Schulman et al. 2015, arXiv:1502.05477. 2nd-order; Engstrom 2020 shows it ≈ PPO-clip when details are matched. **Reject** for implementation cost.

### Tier 2 — Implementation details (Engstrom 2020 + ICLR Blog 2022)

These should be **audited against our scratch code**; many are off by default:

- **Orthogonal init** — gain √2 hidden / 0.01 policy head / 1.0 value head. **Critical.** Check `src/a2c.py::ActorCritic.__init__`.
- **Adam ε = 1e-5** (not torch default 1e-8). DDQN uses 1.5e-4; A2C/PPO almost certainly default. **~1 LOC.**
- **Value-loss clipping** — currently OFF. Andrychowicz 2021 (arXiv:2006.05990) says can hurt; Engstrom says small gain. **A/B test.**
- **Linear LR annealing to 0** over total timesteps. **~5 LOC.** Often >5% return boost.
- **Observation normalization** — running mean/std at PPO wrapper level (different from feature z-score we rejected). Andrychowicz 2021 ranks this top-3 high-impact. **~30 LOC.**
- **"What Matters in On-Policy RL"** — Andrychowicz et al. 2021, arXiv:2006.05990. 250k+ runs ablation; **required reading** before HPO.

### Tier 2 — Recurrent PPO

- **Recurrent PPO (LSTM/GRU)** — Pleines et al. 2022 "Generalization, Mayhems and Limits in Recurrent PPO," arXiv:2205.11104 `[library]`. The canonical "how to code R-PPO correctly" paper: forward-pass shape, sequence arrangement, episode-start hidden-state, **masking padding in the loss** (their Fig 1 — without masking, CartPole-POMDP fails). **~200 LOC in `ppo_recurrent.py`, 1–2 days.** Reference impl `MarcoMeter/recurrent-ppo-truncated-bptt` is a usable template.
- **R2D2 hidden-state burn-in** — Kapturowski et al. 2019, ICLR `[library]`. Burn-in a few steps before computing loss; for stale stored states across PPO epochs.

### Trading-specific PPO tricks

- **Sparse end-of-episode reward** — Lin & Beling 2020 "An End-to-End Optimal Trade Execution Framework based on PPO," IJCAI 2020. Reward = `final_equity − baseline_equity` only at EOD, not per-step. They argue per-step shaped rewards are too noisy on M1. **~20 LOC env change, 2 h.** Directly applicable to our R2/R4 collapse.
- **DeepScalper hindsight + vol aux task** — Sun et al. 2022 (same as DDQN/A2C section).
- **Action shielding / CMDP-PPO** — Borjigin & He 2025 "Safe and Compliant Cross-Market Trade Execution," arXiv:2510.04952 `[library]`. Runtime "shield" projects unsafe actions into a feasible set; cap bar-to-bar position changes to combat over-trading. **~40 LOC env wrapper, 3 h.**
- **Volatility-scaled position reward** — Zhang, Zohren & Roberts 2020 "Deep RL for Trading," arXiv:1911.10107 `[library]`. Reward = position × vol-scaled return. Naturally caps reward magnitude.

### Rejected for PPO

- **TRPO** — Engstrom 2020 shows PPO ≈ TRPO with proper impl.
- **Offline RL on top of PPO (CQL/IQL)** — Kumar 2020, Kostrikov 2022. Designed for static datasets; our env is deterministic so we can always generate fresh on-policy data.
- **DAgger** — Ross et al. 2011. Needs an expert.
- **Async PPO** — wall-clock speedup, not stability fix.

---

## 4. Beyond DDQN/A2C/PPO — Alternative algorithm families

The user explicitly removed the "must be DDQN/A2C/PPO" constraint after Exp 0
revealed PPO's structural collapse. The candidates below are not refinements
of our baselines; they are different algorithm classes.

### Tier 1 — Top 5 candidates for this project

1. **SAC-Discrete** — Christodoulou 2019, arXiv:1910.07207. **Fixes:** auto-tuned entropy (target H̄) directly addresses our recurring flat-collapse failure without manual exploration scheduling. Off-policy with replay → sample-efficient on 600-day train. Used in FinRL benchmarks (Liu et al. 2020, arXiv:2011.09607). **~300 LOC, 2 days.**

2. **QR-DQN (Quantile Regression DQN)** — Dabney et al. 2018, arXiv:1710.10044 (AAAI 2018). **Fixes:** learns full return distribution → unlocks **CVaR / risk-sensitive action selection** natively (critical given the env's ruin termination). DeepScalper itself is risk-aware via distribution-style heads. **~150 LOC, 1 day.** Cleanest distributional Q-learning fit.

3. **Decision Transformer** — Chen et al. 2021 NeurIPS, arXiv:2106.01345. **Fixes:** (a) temporal context native (Exp 3 motivation); (b) trains offline on 600 days of pre-collected baseline trajectories — no exploration collapse possible; (c) test-time return-conditioning: "target Sharpe = X". **~600 LOC, new training paradigm.** *Highest payoff if it works; highest implementation cost.*

4. **CQL (Conservative Q-Learning)** — Kumar et al. 2020, NeurIPS, arXiv:2006.04779. **Fixes:** pre-train on trajectory dataset from our 5 baselines (Flat/Long/Short/Random/MACrossover already in `src/baselines.py`) before online fine-tune. Reduces HPO wall-clock since each trial starts from a non-random policy. **~200 LOC + offline dataset builder, 2 days.**

5. **Recurrent PPO with R2D2-style burn-in** — Pleines 2022 (already cited). Satisfies Exp 3 LSTM/GRU spec while keeping the PPO family the team understands. **~200 LOC, 1–2 days.**

### Off-policy actor-critic family

- **SAC-Discrete** — see Tier 1.
- **DDPG / TD3** — Lillicrap et al. 2016 arXiv:1509.02971; Fujimoto et al. 2018 ICML, arXiv:1802.09477. **Skip:** continuous-action only. Revisit only if Exp 2 goes continuous.

### Distributional value methods

- **QR-DQN** — see Tier 1.
- **IQN (Implicit Quantile Networks)** — Dabney et al. 2018, arXiv:1806.06923 (ICML 2018). Strictly stronger than QR-DQN on Atari; learns quantile *function* (not fixed K). **~200 LOC, 1 day.**
- **FQF (Fully Parameterized Quantile Function)** — Yang et al. 2019, NeurIPS, arXiv:1911.02140. Marginal gain over IQN; skip unless IQN wins and we want one more digit.
- **C51** — strictly dominated by QR-DQN/IQN on every dimension.

### Risk-sensitive RL (very relevant given our ruin termination)

- **CVaR / Distortion-Risk QR-DQN/IQN** — Dabney et al. 2018 IQN paper explicitly shows CVaR_α and Wang distortion-measure action selection. **Trivial after QR-DQN/IQN.**
- **DeepScalper auxiliary heads** — Sun et al. 2022, arXiv:2201.09058 `[library]`. Same idea as in the per-algorithm sections.
- **FineFT** — Qin et al. 2025, arXiv:2512.23773 `[library]`. Risk-aware *ensemble* RL for futures. Method is ensembling rather than a new core algo (see Ensembles below).
- **CMDP-PPO** — Borjigin & He 2025, arXiv:2510.04952 `[library]`. Constrained-MDP with Lagrangian drawdown constraint. **Worth A/B'ing as a principled replacement for R4's hand-tuned β·DD penalty.**
- **Mean-Variance Policy Iteration** — Tamar et al. 2012, ICML. Clunky vs CVaR-via-quantiles.

### Offline RL / batch methods

- **CQL** — see Tier 1.
- **IQL (Implicit Q-Learning)** — Kostrikov et al. 2022, ICLR, arXiv:2110.06169. Avoids querying OOD actions entirely (expectile regression on V + advantage-weighted policy extraction). Often beats CQL with less tuning. **~200 LOC, 2 days.**
- **AWAC** — Nair et al. 2020, arXiv:2006.09359. Offline pretrain → online fine-tune. Good fit for "warm-start from baselines" workflow.
- **Decision Transformer** — see Tier 1.
- **BCQ** — Fujimoto et al. 2019, arXiv:1812.02900. **Skip:** superseded by CQL/IQL.

Trading instances of offline / oracle-style learning in our library:
- **iRDPG** (Liu et al. 2020 AAAI, `5587_13_8812` `[library]`)
- **Universal Trading w/ Oracle Policy Distillation** — Fang et al. 2021, arXiv:2103.10860 `[library]`.

### Sequence modeling

- **Decision Transformer** — see Tier 1.
- **Trajectory Transformer** — Janner et al. 2021, NeurIPS, arXiv:2106.02039. Models full `(s,a,r)` trajectory as tokens, plans with beam search. Slower than DT, similar quality. **Skip in favor of DT.**
- **RvS (RL via Supervised Learning)** — Emmons et al. 2022, ICLR, arXiv:2112.10751. Simple MLP conditioned on return-to-go matches DT. Useful **ablation** to test whether the Transformer matters at all on our 16-dim state.
- **xLSTM-DRL** — Sarlakifar et al. 2025, arXiv:2503.09655 `[library]`. Uses xLSTM blocks inside PPO/A2C for stock trading. **Non-Transformer sequence baseline for Exp 3.**

### Model-based RL

- **Dreamer V3** — Hafner et al. 2023, arXiv:2301.04104. Learns RSSM world model + actor-critic in latent imagination. **Concern: market non-stationarity** — learned dynamics decays as regimes drift; the primacy/plasticity-loss papers in `02_rl_stability_and_training/` flag exactly this. **~1500+ LOC. Defer.**
- **MuZero** — Schrittwieser et al. 2020, Nature, arXiv:1911.08265. MCTS over 3 cheap discrete actions per bar at 1379 bars/episode is computationally wasteful. **Skip.**
- **PlaNet** — Hafner et al. 2019, arXiv:1811.04551. Pure planning, no learned policy. Predecessor to Dreamer. **Skip.**

### Hierarchical RL

- **HRPM** — Wang et al. 2021 AAAI `[library]`. Hierarchical RL for portfolio management with execution costs. **Most directly relevant published precedent.** Likely overkill for single-asset M1.
- **Option-Critic** — Bacon, Harb & Precup 2017 AAAI, arXiv:1609.05140. End-to-end-learned options. Plausible mapping: high-level "be in market vs flat" / low-level "long vs short". Marginal at 3 actions; useful if Exp 2 expands to 9.
- **FuN (FeUdal Networks)** — Vezhnevets et al. 2017, arXiv:1703.01161. Heavy manager/worker abstraction.
- **HIRO** — Nachum et al. 2018 NeurIPS, arXiv:1805.08296. Off-policy hierarchical with subgoal relabeling. Continuous, doesn't fit.

### Evolutionary / non-gradient

- **OpenAI ES** — Salimans et al. 2017, arXiv:1703.03864. Black-box, embarrassingly parallel, no gradients. Robust to non-stationary noisy reward. Sample efficiency is poor but our episodes are short (~1379 bars) and we have 600 days.
- **CMA-ES** — Hansen 2016, arXiv:1604.00772. Better for low-dim parameter spaces; worse for full neural nets.
- **MaxAI (Huber 2025)** — `ssrn_5761402` `[library]`. Explicitly **GA-tuned Q-agent for intraday index futures on 1-min bars, live-deployed positive returns**. **Closest published analogue to our setup.** Reward and action design choices directly transferable.
- **PBT (Population-Based Training)** — Jaderberg et al. 2017, arXiv:1711.09846. Automates the per-reward lr tuning we did manually. **Strong fit as HPO replacement.**

### Imitation + RL hybrids

- **BC + RL fine-tune** — simplest. Pre-train via supervised on MACrossoverBaseline trajectories, then PPO fine-tune. **Strong recommendation as a "free" warm-start experiment.**
- **iRDPG** — Liu et al. 2020 AAAI `[library]`. Imitation-Recurrent-DPG with demonstration buffer + BC loss. **Quote that maps to our PPO collapse:** *"random exploration without goals may bring great losses; agent can hardly learn an effective policy without adequate trials."*
- **Oracle Policy Distillation** — Fang et al. 2021, arXiv:2103.10860 `[library]`. Hindsight-optimal "oracle" distills into a real-time agent. Elegant for trading where ex-post optimal action is computable.
- **GAIL / AIRL** — Ho & Ermon 2016 arXiv:1606.03476; Fu et al. 2018 arXiv:1710.11248. **Skip:** we already have a dense reward; adversarial reward learning adds instability with no payoff.

### Ensembles

- **FineFT** — Qin et al. 2025, arXiv:2512.23773 `[library]`. Ensemble of risk-aware agents for futures. **Cheap post-hoc method**: train DDQN/A2C/PPO seeds, average actions at inference. Could be applied to our Exp 0 v2 outputs as-is.

### Rejected with reason (cross-cutting)

- **AlphaZero** — perfect-model requirement; not available for markets.
- **DDPG / TD3** — continuous-action; revisit only if Exp 2 goes continuous.
- **GAIL / AIRL** — well-specified reward already.
- **FuN / HIRO** — designed for long-horizon sparse-reward; our reward is dense per-bar.
- **Trajectory Transformer** — slower than Decision Transformer at similar quality.
- **PlaNet / Dreamer V1-V2** — superseded by V3, which is itself deferred.
- **NEAT** — topology evolution unnecessary at this scale.
- **C51** — strictly dominated by QR-DQN/IQN.
- **FQF** — marginal over IQN.
- **BCQ** — superseded by CQL/IQL.
- **R3 vol-penalized reward** — already rejected in PROPOSAL.md as overlapping with R4.

---

## 5. Suggested ordering of follow-up experiments

If we keep DDQN/A2C/PPO as the three "official" baselines for the paper:

1. **Exp 0.5 anti-collapse rounds** (already in PROPOSAL §8): γ sweep → hindsight bonus → dueling Q + entropy schedule.
2. **Exp 1.5 DDQN component sweep** (HPO-compatible single-component A/Bs against the Exp-0 DDQN winner): PER → n-step n∈{3,5,10} → Munchausen → NoisyNet → QR-DQN.
3. **Exp 1.5 A2C/PPO Engstrom audit**: orth init gain audit → Adam ε → value-loss-clipping A/B → linear LR anneal → return-based reward-scale (PPO).
4. **Exp 3 sequence models** (already in PROPOSAL): R-A2C + R-PPO (Pleines 2022) + xLSTM-DRL (Sarlakifar 2025) as alternatives to plain LSTM. Decision Transformer as a separate offline track.

If the user accepts loosening the algorithm constraint:

- Replace one of {DDQN, A2C, PPO} with **SAC-Discrete** if PPO continues to collapse after the anti-collapse plan.
- Add **QR-DQN** as a fourth baseline specifically to enable CVaR action selection (risk-aware policy is a natural angle for the project's contribution).
- Run **MaxAI-style GA-tuned Q-agent** as a separate baseline (one-shot run; the paper exists and is the closest published analogue).

---

## 6. Provenance

Compiled 2026-05-16 from four parallel sub-agent reviews of `d:\EA\research_papers\`
covering: DDQN family extensions, A2C family extensions, PPO family extensions,
and algorithms beyond DDQN/A2C/PPO. Each sub-agent received the same project
context (Exp 0 v2 results, MDP design, code locations) and was asked to return
markdown with paper citations on every idea.

See **`JOURNAL.md`** 2026-05-16 entry for the Exp 0 v2 collapse symptoms that
motivated this research, and **`PROPOSAL.md`** §5.1 (Exp 0 Results) and §8
(Anti-Collapse Plan) for the canonical experimental plan that consumes these
ideas.
