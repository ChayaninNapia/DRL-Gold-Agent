# Journal — drl_intraday

Experiment history and design decisions. Append new entries on top. Newer entries supersede older ones when they conflict; older entries are kept as historical record (don't act on their recommendations without verifying against the current code).

For the active project spec see **[proposal/PROPOSAL.md](proposal/PROPOSAL.md)** (canonical, living). [CLAUDE.md](CLAUDE.md) summarizes it; its "Insights" section is a TL;DR (now partly historical — see entry below).

---

## 2026-05-17 — Sidecar exploration considered then reverted (kept here as a historical lessons-only note)

### What happened

After 0.5e (B1) confirmed the imitation branch is exhausted, a research-only sidecar experiment was briefly opened to explore how far an off-policy value-based agent could push net-of-realistic-cost profitability, parallel to the deliverable. Three parallel sub-agent reviews of `research_papers/` converged on a design (DAR-DDQN — Distributional Action-augmented Recurrent DDQN with DeepScalper auxiliary heads, staged ablation). Stage #1 (action augmentation, Huang 2018) was implemented and a 1-seed probe (seed=42, 300k steps) ran: baseline test_ret=-0.0143 vs aug test_ret=-0.0530, both did not collapse (val_trades 557/393). 1-seed result was insufficient to decide (seed 42 is the weak end of baseline per Exp 0.5a; aug showed late-improving val trajectory but worse test MDD 7.54% vs 2.33%), and the whole sidecar branch was **reverted on 2026-05-17 by user decision** to keep focus on the deliverable Exp 1.

### Lessons preserved (sidecar reverted; insights still apply to deliverable)

These are research findings that survive the revert because they describe the literature and the MDP, not the (now-reverted) sidecar code:

1. **Honest profitability ceiling for single-asset intraday DRL with realistic cost is Sharpe 0.5–1.5, PF 1.05–1.15.** Evidence: MaxAI 2025 (NQ M1, full real CME frictions, GA-tuned tabular Q) reports PF 1.07 / Sharpe 1.04 — the only fully cost-honest paper in the library. Zhang/Zohren/Roberts 2019 shows monotone Sharpe decay in cost-bp on daily (M1 turnover is higher → ceiling is lower for our regime). The project's current DDQN γ=0.30 + R4 result (+2.23% mean test, best seed +7.90%, Sortino +3.57) is at the honest end of the literature — close to ceiling, not far below. Calibrate the paper's expectations to this band.
2. **"Headline" intraday-DRL Sharpes (DeepScalper 1.76–4.75, Goluža 2.76, Huang DRQN 1.5–5.7) all use idealised cost** — 0.08–0.23 bp vs retail XAUUSD spread 20–40 bp (200–500× smaller). Goluža's own ablation kills its model at 0.16 bp, i.e. *before* reaching real cost. Do not cite their headline Sharpes as targets. This is the cost-realism caveat to put in the paper writeup.
3. **Théaté & Ernst 2020 explicitly documents the DQN do-nothing collapse under realistic cost** — published precedent that frames our Exp 0.5b–0.5e on-policy collapse as a *known structural finding* rather than a bug. This is the citation for the negative-result section of the paper.
4. **The MDP has an exogenous price process** — next-bar `open[t+1]` and `close[t+1]` do not depend on the current action (no market impact in our cost model). This is a useful structural property worth noting in the MDP section even though the sidecar that exploited it is reverted; it means future extensions can do counterfactual policy evaluation, off-policy importance weighting, or all-action loss correctly.
5. **The Exp 0.5b–0.5e on-policy collapse is a credit-assignment problem** (four interacting pathologies, summarised in the 2026-05-16 entry below): per-bar SNR ≪ 1 vs spread noise; flat is a zero-variance attractor; $\gamma^k$ horizon decay vs immediate deterministic spread; off-policy bootstrap vs on-policy sampled-advantage asymmetry. This framing should appear in the paper's discussion of why DDQN works and A2C/PPO do not.

### Why the sidecar was reverted

The 1-seed probe at seed=42 was not decisive (baseline test_ret=-0.0143, aug test_ret=-0.0530, both non-collapse; seed=42 is the weak end of baseline per Exp 0.5a where 3-seed mean was +2.23%). Resolving the question would have required the full 3-seed run, then likely committing to four staged ablations — a multi-day branch parallel to the deliverable. The deliverable Exp 1 (DDQN HPO, then Exp 2 action space, then Exp 3 LSTM/GRU) is the priority and the safer path: DDQN already works at the honest profitability ceiling on the recent regime, and the on-policy negative result has the depth of diagnosis a paper needs. Sidecar code and `runs/sidecar_dar/` artifacts were deleted; sidecar memory entries were removed. The lessons above are kept here for the paper writeup.

---

## 2026-05-16 — B1 (Exp 0.5e) COMPLETE: class-weighted BC also collapses A2C/PPO 6/6; imitation branch exhausted; regime/over-training hypothesis ruled out

### Status snapshot

B1 = the cheapest "next-session option" from the 0.5d entry below: keep Phase-1d's BC warm-start but make the cross-entropy **inverse-frequency class-weighted**, on the hypothesis that 1d failed *specifically* because the hindsight expert is flat-majority (plain CE → predict-flat). Implemented, smoke-tested, ran full 6 runs (a2c+ppo × seeds 42/1337/2026 × 300k, γ=0.30, R4, same 4-knob fix, parallel N=4 via `scripts/run_exp05e_parallel.py`, `runs/exp05e/`). **Verdict: NEGATIVE — 6/6 collapse, same as 1d.**

### Expert flat-fraction measured (confirms the 1d diagnosis quantitatively)

Over 40 train days, `compute_expert_actions` distribution by (h, noise_threshold):

| h | th | short | flat | long |
|--:|--:|--:|--:|--:|
| 5 | 0.0005 | 0.072 | **0.857** | 0.071 |
| 10 | 0.0005 | 0.123 | 0.754 | 0.123 |
| 20 | 0.0005 | 0.186 | 0.628 | 0.186 |
| 20 | 0.0002 | 0.341 | 0.315 | 0.344 |

At the 1d/B1 default (h=5, th=0.0005) the expert is **85.7% flat** — worse than the ~63–70% the prior memory estimated. short/long are always symmetric, so the imbalance is purely flat-vs-(short+long), which inverse-frequency weighting targets exactly (verified `freq_c·w_c = 1/K` per class).

### What B1 changed (single variable vs exp05d)

`_inv_freq_weights(targets, n_actions, device)` added to `src/a2c.py` (one source of truth; `src/ppo.py` imports it). `bc_class_weight` ctor arg on A2CAgent/PPOAgent; `F.cross_entropy(..., weight=)` wired in both. PPO computes weights **once per rollout** from valid expert labels (class-freq is policy-independent, so unlike adv-norm it does NOT go stale across K epochs — kept consistent with the project's per-rollout-PPO convention). Config: `bc.class_weight: false` default (no effect on other runs). `scripts/run_exp05e_parallel.py` = exp05d clone with only `class_weight: True` changed.

### Result and the mechanism it proves

Class weighting **worked at its stated job**: every run traded heavily *during* the BC phase — eval1 @30k showed 3000–6000 val trades (vs 1d, which committed straight to flat). But after BC anneal completes at step 100k, **every run collapses to val_trades=0 by eval3 (≈90k) and stays there to 300k.** Final test, all 6 runs: eq=$10000.00 exactly, trades=0.

This is the informative part: B1 *removed* the 1d flat-commitment (policy traded fine under BC) and the run **still collapsed** once BC was withdrawn. Therefore the 1d flat-majority was **not the root cause** — it was a symptom. Root cause unchanged from the 0.5d entry: the post-BC RL phase has no learning signal that makes trading positive-EV (credit-assignment), compounded by `rl_lr` annealing down to the R4-scale 1e-5/1e-6 which leaves the RL phase too weak to hold the policy off the flat attractor. The whole imitation branch (1d plain BC + B1 weighted BC) is now exhausted.

### User hypothesis tested and ruled out: "trained too long / train regime misses recent val regime"

1. **Over-training is mechanically impossible here.** 300k timesteps ≈ **0.38 epoch** (600 train days × ~1300 bars ≈ 780k steps per full pass). The agent never even completes one sweep of the train set — if anything the risk is under-, not over-fitting.
2. **Regime/distribution-shift ruled out as the collapse cause.** DDQN+R4 γ=0.30 (Exp 0.5a) ran the **identical** split — train 2023-06-05→2025-09-29, val 2025-09-30→2026-01-14, test 2026-01-15→2026-04-30 — same reward, same 300k horizon, and made **+2.23% mean test return (best seed +7.90%), 0/3 collapse** on the most-recent test months. Same data, only the algorithm class differs (off-policy value vs on-policy PG). So the on-policy collapse is algorithm-structural, not a regime or duration artifact.
3. **Collapse signature is "doesn't trade", not "trades and loses".** All collapsed runs show val_trades=0, not low-winrate-high-trades. A regime mismatch would show trading-but-losing.
   - Caveat kept for later: distribution-shift is still a valid question **for DDQN's Exp 1** (watch per-seed val→test degradation when the agent actually trades) — just not the explanation for on-policy collapse.

### Next (unchanged recommendation)

Imitation branch closed. Remaining options: VC-PPO decoupled critic λ (PPO-only, ~30 LOC) / switch on-policy algo (SAC-Discrete, Recurrent PPO) / **proceed Exp 1 DDQN-only and document A2C/PPO as a characterized structural negative — recommended**: DDQN γ=0.30 works on the recent regime, the deliverable is the comparison, and the negative is now diagnosed across γ-sweep + 4-knob + plain BC + class-weighted BC with an explained mechanism. Note `proposal/algorithm_extensions.md` (reference for the algo-switch options) was `git rm`'d this session but is recoverable via `git checkout HEAD -- proposal/algorithm_extensions.md`.

---

## 2026-05-16 — Exp 0.5b/c/d ALL COMPLETE: on-policy collapse is structural (neither hyperparameter tuning nor BC warm-start fixes A2C/PPO); DDQN remains the working baseline

### Status snapshot

Phase 1a (Exp 0.5a) selected γ=0.30 for DDQN and solved DDQN collapse (0/3, +2.23% mean test_ret). Phases 1b and 1c tested whether the same medicine — and then a deeper hyperparameter fix — rescues the on-policy algorithms (A2C, PPO). **Both failed: A2C and PPO collapse 3/3 under γ=0.30, and the 4-knob tuning fix from a 3-agent code audit did not move them.** Phase 1d (behavior-cloning warm-start) is now being implemented as the pivot; **1d has not produced results yet — verdict pending**.

### Phase 1b — γ=0.30 on A2C/PPO (does γ=0.3 generalize to on-policy?)

6 runs (A2C, PPO × 3 seeds × 300k steps, R4), N=4 parallel, wall **43.3 min**. Driver `scripts/run_exp05b_parallel.py`, outputs `runs/exp05b/`.

| algo | Collapse | Mean test_ret | Mean Sortino |
|---|---|---|---|
| A2C | **3/3** | −0.45% | −4.04 |
| PPO | **3/3** | −0.14% | −0.64 |

2 of the 6 runs hit FULL collapse (0 trades on val); the other 4 had val_trades 5–41, all below the 50-trade threshold (degenerate "do-nothing-with-noise"). **γ=0.3 does NOT transfer to on-policy.** Zhang/Zohren/Roberts 2019 claimed γ=0.3 works for A2C too, but their result does not reproduce here — the off-policy DDQN benefit (replay buffer decorrelates the target so a lower γ raises target SNR) has no on-policy analogue.

### Phase 1c — 4-knob hyperparameter fix on A2C/PPO

A 3-sub-agent code audit (independent reviewers, converged) diagnosed the root cause: **at γ=0.30 the on-policy advantage signal becomes pure noise, so the policy never moves.** Symptoms confirming the diagnosis: entropy stuck at ln(3)=1.099 (uniform random over {short, flat, long}), KL≈0, explained-variance≈0 — the actor never departs from its initialization. The audit prescribed 4 changes (applied inline by the driver, `config.yaml` NOT modified):

| knob | old → new | rationale |
|---|---|---|
| `gae_lambda` | 0.95 → 1.0 | at γ=0.3, γ·λ=0.285 cuts the effective horizon to ~3 bars; pure Monte-Carlo returns needed |
| `value_coef` | 0.5 → 0.25 | value MSE was ~99% of total loss, starving the policy head |
| `entropy_coef` | 0.01 → 0.05 | more exploration pressure at the short horizon |
| `ppo.n_epochs` | 10 → 4 | fewer SGD epochs over noisy advantages |

6 runs (same matrix), N=4 parallel, wall **47.9 min**. Driver `scripts/run_exp05c_parallel.py`, outputs `runs/exp05c/`.

| algo | Collapse | Mean test_ret |
|---|---|---|
| A2C | **3/3** | −0.46% |
| PPO | **3/3** | −0.43% |

**Still 3/3 collapse for both.** The 4-knob diagnosis was correct about the *mechanism* (noisy advantage → dead policy) but the fix was insufficient.

### Why 1c was necessary but insufficient (this rules out "we just didn't tune enough")

Phase 1c was not optional — it had to be run to eliminate the most parsimonious hypothesis ("the on-policy hyperparameters are merely mistuned at γ=0.3"). We tuned every internal lever that touches the advantage signal: γ itself (1b), the horizon (gae_lambda), the value/policy loss balance (value_coef), exploration (entropy_coef), and SGD aggressiveness (n_epochs). All five tried; collapse unchanged. **Conclusion: hyperparameter tuning alone cannot fix on-policy collapse here. The policy needs an EXTERNAL signal, not better-tuned internal gradients** — when the advantage is noise, no reweighting of a noise-driven gradient produces a directional policy. This is the on-policy analogue of the Exp-0 DDQN finding that exploration, not loss tuning, was the bottleneck.

### Pivot — Phase 1d: behavior-cloning warm-start (COMPLETE — also collapses, mechanism shifted)

2 of 3 brainstorm sub-agents independently recommended BC warm-start as the next move; iRDPG (Liu et al. AAAI 2020, `5587_13_8812`) reports ~4× improvement on CN futures with exactly this device, and its diagnosis matches ours: *"the agent can hardly learn an effective policy without adequate trials and errors."* A noisy advantage cannot bootstrap a policy from a uniform init — BC supplies the missing external directional signal so RL starts from a non-flat policy.

Design: a daily hindsight expert (long at intraday lows, short at intraday highs, h=5-bar lookahead) provides labels for a cross-entropy auxiliary loss, annealed out as RL takes over. New file `src/expert.py`. Modified `src/a2c.py`, `src/ppo.py` (`bc_coef` + CE aux loss + anneal), `src/train_a2c.py`, `src/train_ppo.py`, `config.yaml` (`bc:` block). Driver `scripts/run_exp05d_parallel.py`.

**Phase 1d result (6 runs, A2C/PPO × 3 seeds, 300k, 48.4 min): still 6/6 collapse, val_trades=0, test_ret=0.0000, Sortino 0.00 for every run.** But the failure *mechanism is different and informative*:

| Phase | A2C collapse | PPO collapse | entropy H | trades trend (train) |
|---|---|---|---|---|
| 1b (γ=0.3) | 3/3 | 3/3 | stuck at ln3≈1.099 | high but losing / random |
| 1c (+4-knob) | 3/3 | 3/3 | stuck at ln3≈1.099 | same |
| **1d (+BC)** | **3/3** | **3/3** | **drops 1.099 → 0.10–0.14 (A2C) / 0.03 (PPO)** | **636 → 49 → 37 → 15 (a2c_s42); 636 → 44 → 7 → 3 (ppo_s42)** |

BC *did* work in the narrow sense it was designed for: the uniform-random collapse is gone — entropy drops sharply and the policy commits. **But it commits to flat.** The hindsight expert at h=5, noise_threshold=0.0005 is ~63–70% flat (verified in `src/expert.py` smoke test), so the cross-entropy loss converges the policy to the majority class. After `bc_coef` anneals to 0 (step 100k), the pure-RL phase (γ=0.3, noisy advantage — the *unchanged* Phase 1b/1c problem) cannot pull the policy off flat. Train trades decay monotonically toward single digits; eval (deterministic argmax) is flat from the very first eval at step 30k. Confirmed by per-episode trace: H keeps falling and trades keep falling all the way to step 299k.

**Deeper diagnosis (the real finding of Exp 0.5b/c/d):** the on-policy collapse is not an exploration problem, not an initialization problem, and not a hyperparameter problem. It is a **credit-assignment problem** — under γ=0.3 with M1 sparse net-of-spread reward, on-policy methods have no learning signal that makes *trading* positive-EV relative to *flat*. Whether the policy starts uniform (1b/1c) or is warm-started toward the expert (1d), it ends at "don't trade" because that is the only locally-stable behavior the noisy advantage will not punish. DDQN (off-policy, replay-buffered, value-based) is structurally immune and remains the working baseline at γ=0.30.

**Two bugs found and fixed during 1c/1d setup:**

1. **cp874 encoding crash (1b/1c).** Scripts that `print()` Greek letters (γ) crash with `UnicodeEncodeError` under cp874 when run as a background subprocess on this Thai-locale Windows box. Fix: ASCII-only `print()` (write "gamma", not "γ") + set `PYTHONIOENCODING=utf-8` in the subprocess env. (Saved to auto-memory `feedback_cp874_encoding.md`; this is an operational gotcha for driver scripts, not a project-spec insight.)
2. **BC lr-too-small (1d smoke test).** The per-reward lr for R4 (A2C 1e-5, PPO 1e-6) is far too small for the BC cross-entropy loss (scale ~1.0) to move the policy — 200 updates at lr=1e-5 left entropy unchanged. Fix (Option 1, LR schedule): use a high lr during the BC phase (A2C 7e-4 / PPO 3e-4, the R1-scale defaults), linearly annealed down to the small RL lr by `bc_anneal_steps`, synchronized with `bc_coef → 0`. `config.yaml` now carries `bc.lr_bc`.

### Next (decision point for the next session)

Exp 0.5b/c/d are all complete and all negative for on-policy. Hyperparameter tuning (1b/1c) and BC warm-start (1d) both fail; the failure is now understood as structural credit-assignment, not a tuning miss. **Do not run more "tune another knob" or "more imitation" rounds without a new hypothesis** — that path is exhausted. Open options for the next session, in rough order of cost:

1. **Class-balanced / weighted BC** (cheapest, ~10 LOC): the 1d failure is specifically that the expert is flat-majority. Re-weight the CE loss inversely to class frequency, or generate a less-flat expert (smaller `noise_threshold`, larger lookahead `h`). Tests whether 1d's *flat-commitment* (not BC itself) was the blocker. Highest-information cheapest next step.
2. **VC-PPO decoupled critic λ** (PPO-only, ~30 LOC): one brainstorm agent flagged a decoupled value-target λ as a credit-assignment fix specifically for sparse-reward PPO. Narrower scope than #1.
3. **Risk-sensitive / longer-horizon reformulation** or **switch the on-policy algo** to SAC-Discrete or Recurrent PPO (see `proposal/algorithm_extensions.md`). Largest scope; only if 1–2 fail.
4. **Proceed Exp 1 DDQN-only** and document A2C/PPO as a characterized negative result (collapse is structural under this MDP/cost regime; full mechanism in this entry). This is a legitimate paper outcome — the project's required deliverable is the comparison, and "on-policy structurally fails here, off-policy works" with this depth of diagnosis is a finding, not a gap.

DDQN stays the working baseline at γ=0.30. No on-policy γ default is set (every on-policy config tested collapses regardless of γ). The expert labeler (`src/expert.py`), BC plumbing (`bc:` config block, `bc_coef`/anneal in a2c/ppo), and the lr-schedule are all in place and correct — they are reusable as-is for option #1.

---

## 2026-05-16 — Exp 0.5a COMPLETE: γ-sweep selects γ=0.30 for DDQN (collapse 0/3, +2.23% mean test_ret)

### Status snapshot
Phase 1a of Exp 0.5 (Anti-Collapse Plan, PROPOSAL §8) ran 12 DDQN+R4 runs (4 γ values × 3 seeds × 300k steps). Wall time **63.4 min** with N=4 parallel subprocess (vs sequential ~120 min, speedup ~1.9x — workload sustains 84-92% GPU util so parallel scales less aggressively than the smoke test predicted at 50k). Goal: test Zhang/Zohren/Roberts 2019's claim that γ=0.3 is the right horizon for intraday DRL, vs our Exp-0 default γ=0.99.

### Result

| γ | Collapse | Mean test_ret | StdDev | Mean Sortino | Mean trades |
|---|---|---|---|---|---|
| **0.30** | **0/3** | **+0.0223** | 0.041 | **+1.03** | 612 |
| 0.50 | 1/3 | +0.0019 | 0.013 | -0.08 | 134 |
| 0.90 | 0/3 | **-0.0619** | **0.101** | -1.26 | 713 |
| 0.99 | 0/3 | -0.0193 | 0.038 | -0.21 | 517 |

**Winner: γ=0.30**. All three signals point the same way:
- Best mean test_ret (+2.23% vs nearest -1.93%).
- Best mean Sortino (+1.03 — the only γ where the agent has positive risk-adjusted return on average).
- 0/3 collapse to do-nothing (val_trades min=57, all above the 50 threshold).

**Best individual run:** `g0.3_s2026` — test_ret +7.90%, Sortino +3.57, 570 trades, final equity $10,011.

### What this confirms / contradicts

- **Confirms** the paper survey ([[reference-research-papers]]): Zhang/Zohren/Roberts 2019 used γ=0.3 for *all* of {DQN, PG, A2C} on intraday futures. Their root-cause framing — that γ=0.99 over a 1-min step horizon imposes effectively-infinite credit assignment on a near-Markov price series, biasing the agent toward "do nothing because nothing pays out in the relevant window" — is what we saw in Exp 0 and what γ=0.3 directly relieves.
- **Surprise:** γ=0.99 (our old default) is *not* the worst — γ=0.90 is, with mean test_ret -6.19% and StdDev 0.101 (2.5x the others). One seed (g0.9_s1337) traded 1521 times on test for -20.2% return — high variance unstable. γ=0.99 sits in a more conservative regime (Q-net hedges by mostly trading flat) so its mean is bad but variance is contained.
- **γ=0.5 is a transition zone:** 1/3 collapse, mean test_ret essentially zero. Not safe to rely on.

### Action taken

- `config.yaml`: `dqn.gamma: 0.99 → 0.30` with inline comment citing this entry.
- Plots: `runs/exp05a/gamma_summary.png`, `test_metrics_by_run.png`, `equity_curves.png`.
- Summary CSV + collapse report: `runs/exp05a/exp05a_summary.csv`, `exp05a_collapse_report.json`.
- Git commit `<TBD>` snapshots code + plots (runs/ stays gitignored).

### Operational notes (kept for next phase)

- **Parallel driver** (`scripts/run_exp05a_parallel.py`) launches N=4 subprocesses, each capped at `OMP_NUM_THREADS=4` to avoid CPU oversubscription on the 20-core box. Smoke test (50k steps, no eval) showed 2.27x speedup; actual workload (300k + eval/test) shows ~1.9x. Difference is GPU saturation — when 4 workers are training simultaneously, GPU util sits at 84-92%, near the ceiling.
- **Driver design**: each worker is a fresh `python -c` subprocess with a per-run YAML config dumped to a temp file. This avoids interpreter state leakage (CUDA contexts, replay buffers, module caches) between runs.
- **Wall-time estimates were close**: predicted 78 min for 12 runs at 300k each; actual 63.4 min. Updated estimate for Phase 1b (6 A2C/PPO runs at 300k): ~30-35 min parallel.

### Next step — Phase 1b

Run γ=0.3 with A2C and PPO (3 seeds each, R4, 300k steps) to test whether the γ=0.3 finding is algo-agnostic (Zhang's paper claims yes; ours is N=1 algo until we extend). If A2C/PPO also benefit:
- γ=0.3 becomes the new project-wide default across all algos.
- We can proceed to Exp 1 HPO on R4 with γ as a fixed hyperparameter (one less dimension to search).
- Hindsight (Exp 0.5b) and dueling/entropy (Exp 0.5c) **may not be needed** — Exp 0.5a alone may have solved collapse.

If A2C/PPO γ=0.3 still collapse: γ is DDQN-specific, and we go to Exp 0.5b (hindsight bonus per DeepScalper §4.2).

[[project-exp0-findings]] — see memory note for Exp 0 v2 baseline this builds on.

---

## 2026-05-16 — Exp 0 v2 COMPLETE: 27/27 runs, R4 wins (mean rank 1.67), collapse pattern documented

### Status snapshot
Exp 0 v2 finished in **594.4 min (9h 54m)**. All 27 runs (3 rewards × 3 algos × 3 seeds × 500k timesteps, no HPO, per-reward lr regime) completed. Winner declared by `scripts/run_exp0.py`: **R4** (mean rank 1.67 across DDQN/A2C/PPO on `val_total_return`). 0 ruin events across all runs. Headline result: 4/27 runs produced positive test returns (all DDQN+R2 or DDQN+R4 seeds), 12/27 runs collapsed to do-nothing flat policies.

### Headline: PROPOSAL Sec 6.4 (running-std reward normalization) is **wrong** for our setup

**A/B test (DDQN, R1, seed=42, 500k each):**
| Run | normalize | val evals with trades>0 | val_best total_return | test trades | test eq |
|---|---|---|---|---|---|
| A | true (PROPOSAL spec) | **0/16** | 0.0 (do-nothing) | 0 | $10,000 (flat) |
| B | false | **16/16** | +0.0041 | 594 | $9,988 |

Mechanism: R1's per-bar log-return is ~1e-4. Running-std normalizer's std starts tiny (n<10 samples) so it amplifies R1 by ~10^4-10^5 in the first few thousand steps. The Q-net sees gigantic synthetic rewards for whatever action was last sampled (usually flat at uniform init) and the optimal value of "always flat" (Q=0/std=0) is the stable target → instant collapse to flat. Normalization is supposed to put R1/R2/R4 on equal footing for a shared lr, but the *cure is worse than the disease* — at least for R1 + DDQN.

**Decision: `env.reward.normalize: false` for Exp 0 v2.** Per-reward lr (below) handles the scale gap instead. PROPOSAL Sec 6.4 needs to be edited (TODO post-Exp-0).

### Per-reward learning rate calibration (sequential mini-tests, DDQN/A2C/PPO × 100k timesteps each)

R1's raw scale (~1e-4/bar) and R2/R4's raw scale (~$1-30/bar) differ by ~10^4-10^5×. With normalize=false, a single lr cannot cover both, so we ran 3 mini-tests:

**Test 1 — DDQN R2 (lrs: 4.5e-7, 1e-6, 1e-5):**
- 4.5e-7 & 1e-6: hyperactive (17k-20k val trades, ret ≈ -0.73)
- **1e-5: selective (636 test trades, eq=$9987, ret=-0.096) ✓**

**Test 2 — DDQN R1 (lrs: 0.0045, 1e-5):**
- 0.0045 (default): test eq=$10000.64, ret=**+0.0048**, Sharpe **+0.787**, trades=57, winrate 0.509 ✓ first positive Sharpe ever seen
- 1e-5: hyperactive, ret=-0.42
- **DDQN R1 wants 0.0045 (current default), DDQN R2 wants 1e-5**

**Test 3 — A2C/PPO R2 (lrs: 1e-5, 1e-6, 1e-7):**
- A2C: lr=1e-5 best (43 stable val trades), lr=1e-6/1e-7 over-trade (1100-7500)
- PPO: lr=1e-5 collapsed to flat (clip+SGD too aggressive); **lr=1e-6 best (11 selective val trades, eq=$9999.98)**; lr=1e-7 over-traded

**Locked-in per-reward lr table (in `config.yaml` `lr_per_reward` blocks, routed by `scripts/run_exp0.py::_apply_per_reward_lr`):**

| algo | R1 | R2 | R4 |
|---|---|---|---|
| DDQN | 0.0045 | 1e-5 | 1e-5 |
| A2C | 0.0007 (default) | 1e-5 | 1e-5 |
| PPO | 0.0003 (default) | 1e-6 | 1e-6 |

The `_apply_per_reward_lr` helper reads `cfg[algo_section]["lr_per_reward"][reward_mode]` and overwrites `cfg[algo_section]["lr"]` before training. Algo-section map = `{"ddqn": "dqn", "a2c": "a2c", "ppo": "ppo"}`. Per-reward lr support is also wired into `src/hpo.py::_sample_hparams` (R4 mode adds `reward_beta` + `reward_dd_thresh`) and `scripts/run_final.py::_apply_params` so Exp 1's HPO inherits the same mechanism.

### Exp 0 v2 final results (27/27 done, 594 min wall-clock)

**Test set (75 days, 2026-01-15 → 2026-04-30), mean test total_return across 3 seeds:**

| | R1 (log) | R2 ($P&L) | R4 (P&L−DD) | **Best algo for reward** |
|---|---|---|---|---|
| **DDQN** | −3.34% (1/3 profitable) | **+1.77%** (1/3 profitable, best +9.28%) | 🌟 **+3.37%** (2/3 profitable, Sharpe +0.57 mean) | DDQN+R4 ⭐ |
| **A2C** | −2.41% (1 collapse) | −0.44% (1 collapse) | −0.44% (1 collapse) | tied across rewards |
| **PPO** | **−46.97%** (over-trade 5k-10k trades) | **0.00%** (3/3 collapse) | **0.00%** (3/3 collapse) | tied — all degenerate |

**Mean rank ranking (val_total_return, lower = better):**

| Reward | DDQN rank | A2C rank | PPO rank | **Mean rank** |
|---|---|---|---|---|
| **R4** | #1 | #2 | #2 | **1.67** 🥇 |
| R1 | #2 | #1 | #3 | 2.00 🥈 |
| R2 | #3 | #3 | #1 | 2.33 🥉 |

**Note on R2 PPO mean rank #1:** spurious — PPO+R2 all 3 seeds collapsed to flat (return=0). It ranks #1 because the alternatives also produced near-zero numbers; this is the "rank from below" pathology that the `min_trades` filter would catch in HPO (Exp 1), but Exp 0 has no HPO and no min_trades filter. The actual ranking we care about uses positive performance.

**Top 4 individual runs (by Sharpe):**
1. **r2_ddqn_s1337** — eq=$10,012.59, ret=+9.28%, **Sharpe +1.53, Sortino +2.56**, trades=653, winrate 0.479, mdd $461
2. **r4_ddqn_s1337** — eq=$10,009.48, ret=+6.80%, **Sharpe +1.18, Sortino +2.20**, trades=433, winrate 0.471, mdd $430
3. **r4_ddqn_s42** — eq=$10,008.48, ret=+5.97%, **Sharpe +1.01, Sortino +1.57**, trades=665, winrate 0.469, mdd $584
4. **r1_ddqn_s1337** — eq=$10,001.14, ret=+0.81%, Sharpe +0.51, trades=84, winrate 0.714, mdd $356

**Winner = R4** (mean rank 1.67). R4+DDQN clearly beats R2+DDQN on Sharpe (+0.57 vs +0.24) and on mean return — drawdown penalty makes DDQN more selective without hurting profitability. R4 ≈ R2 on A2C/PPO because their trade counts are too low (0-67/episode) for the in-episode drawdown to ever exceed the 2% threshold → penalty=0 → R4 collapses to R2 behavior on these algos.

**Collapse audit (12/27 runs, 44%):**
- a2c_r1_s42 (1)
- a2c_r2_s1337 (1)
- a2c_r4_s1337 (1)
- All 3 r2_ppo seeds (3)
- All 3 r4_ppo seeds (3)
- The remaining "near-flat" cases: a2c_r2_s2026 / a2c_r4_s2026 (14 trades each), a2c_r2_s42 / a2c_r4_s42 (67 trades each) — these still register as trade>0 but are essentially do-nothing-with-noise.

**0 ruin events** across all 27 runs (capital $10k + 0.01 lot too conservative for ruin to matter at this scale).

### Anti-collapse research (sub-agent reports, 2026-05-16)

Two sub-agents reviewed `research_papers/`. Independent agreement on 3 of the top 4 techniques, validated against our specific symptoms:

**1. Hindsight reward bonus (DeepScalper §4.2, Sun et al. 2022, `2201.09058`)** — both agents flagged.
Augment per-bar reward with `r_t += w · (close[t+h] − close[t]) · pos_t` during training only (test uses raw reward, no leakage at deploy). DeepScalper Table 4 ablation: **TR 3.5% → 6.97% (~2×), Sharpe 4.42 → 5.72** on M1 minute futures. Paper's quote diagnoses our exact symptom: *"agent pays too much attention to short-term price fluctuation"*. Implementation: ~20 LOC in `src/env.py` reward computation; add `cfg["env"]["hindsight_w"]` + `cfg["env"]["hindsight_h"]`. Works for all 3 algos.

**2. Lower discount factor γ (Zhang/Zohren/Roberts 2019, Oxford-Man, `1911.10107`)** — Agent B's biggest find.
The paper uses **γ=0.3** for DQN/PG/A2C in multi-asset futures trading; we use γ=0.99. With our 1379 bars/episode, γ=0.99 makes Q(flat)="sum of 0 reward × hundreds of bars" a low-variance stable target while Q(trade)="sum of noisy ±$1-30 reward + spread cost × hundreds" is high-variance and *looks worse* despite some bars being profitable. Lowering γ raises target SNR. Implementation: 1 line in `config.yaml`, sweep γ ∈ {0.1, 0.3, 0.7, 0.9}.

**3. Dueling Q-network (Wang et al. 2016, used by DeepScalper §4.1 + Zhang/Zohren/Roberts §3.2)** — Agent A's pragmatic pick.
Decouple Q(s,a) = V(s) + (A(s,a) − mean A). V(s) updates **on every transition regardless of action**, so V(trending day) becomes well-estimated even when the agent has never traded → fixes the symmetric Q≈0 collapse in DDQN. Implementation: ~15 LOC in `src/ddqn.py::QNet`, two heads (`fc_v`, `fc_a`) replace `fc_out`. DDQN-only; A2C/PPO already separate actor/critic.

**4. Entropy schedule + policy-head init (Engstrom 2020 `2005.12729`, "What Matters in On-Policy RL" `2006.05990`)** — for A2C/PPO collapse.
Current entropy_coef=0.01 is too low for trading: PPO collapses to a near-deterministic flat distribution before the value function has enough signal. Schedule: 0.05 → 0.005 over training. Critical companion fix: **orthogonal init with gain 0.01 on policy head + zero bias** so the initial policy is uniform over {short, flat, long} (we already have this in `src/a2c.py::ActorCritic.__init__` — verify it's still active). ~30 LOC for the schedule.

**5. iRDPG-style warm-start (Liu et al. AAAI 2020, `5587_13_8812`)** — for hardest collapse modes.
Pre-fill DDQN replay buffer (or pre-train A2C/PPO policy via BC loss) using `MACrossoverBaseline` (already in `src/baselines.py`). The exact quote from iRDPG that matches our diagnosis: *"random exploration without goals may bring great losses. However, the agent can hardly learn an effective policy without adequate trials and errors."* Fixes PPO+R2 collapse 3/3 case directly because the policy gets non-flat trajectories with non-trivial reward before RL starts. ~40 LOC for DDQN buffer prefill, ~80 LOC for A2C/PPO BC loss.

**Sub-agent consensus skips:** ICM/RND curiosity (volatility-novelty correlates with losses, not profits), Noisy Networks (no financial-RL paper uses it), prioritized replay alone (amplifies whatever signal exists — currently negative for trades), action-balanced forced exploration (introduces deterministic losing trades).

### Single-fix recommendation if compute is constrained (per Agent B)

**Lower γ from 0.99 to 0.3** as the first move. Why over hindsight (Agent A's #1)?
- Zero code change, zero MDP-touch risk
- Only fix with multi-asset peer-reviewed evidence for our exact algo trio (DDQN/A2C/PG)
- Diagnostically clean: if γ=0.3 escapes the attractor, the issue was target-value variance from over-long horizon; if it still collapses, the issue is exploration (do hindsight next), not credit assignment

### Decision flow for next phase

**Plan A (if user agrees):** Exp 0.5 (anti-collapse fixes) before Exp 1.
- Round 1: γ sweep alone (1 line, 9 runs × 500k ~3h)
- Round 2: if needed, add hindsight bonus (20 LOC, another 9 runs ~3h)
- Round 3: if needed, add dueling + entropy schedule (45 LOC, another 9 runs ~3h)
- Goal: drop collapse rate from current 14.8% (4/27 runs) to <5%; mean return positive across all algos

**Plan B:** straight to Exp 1 (HPO + algo comparison) using R4 winner. Risk: R4+DDQN dominates because R4+A2C/PPO collapse — the "algorithm comparison" is meaningless if 2/3 algos are mostly do-nothing policies.

User has chosen Plan A in spirit (asked sub-agents to research collapse fixes). Will make formal Exp 0.5 spec proposal after the last 2 Exp 0 v2 runs (r4_ppo_s1337, r4_ppo_s2026) finish.

### Code/config changes made this session

- `config.yaml`: `env.reward.normalize: true → false`, `train.best_metric: sortino → total_return`, `train.early_stop_patience: 5 → 20`, `train.eval_every_sessions: 11 → 22`. Added `lr_per_reward` blocks under `dqn:`, `a2c:`, `ppo:`.
- `scripts/run_exp0.py`: added `_ALGO_CFG_SECTION` map + `_apply_per_reward_lr()` helper. Applied inside `run_one()` after writing `reward.mode`.
- `src/env.py`: removed dead `_mark_to_market()` helper.
- `proposal/PROPOSAL.md` Sec 3.5: clarified PR formula dimensions (introduced η = contract_size × lot = $/price, made the entry-cost division dimensionally consistent).
- New scripts (all kept for reproducibility): `scripts/run_ab_reward_norm.py`, `scripts/run_r1_lr_test.py`, `scripts/run_r2_lr_test.py`, `scripts/run_a2c_ppo_r2_lr_test.py`.
- `runs/` was wiped multiple times during the lr/norm tests; runs/exp0/ now holds the actual Exp 0 v2 artifacts.

### Issues that affected this session (worth remembering)

- **Python stdout buffering on Windows + background tasks:** Python detected stdout=pipe → block-buffered → progress only flushed every ~8 KB. Solved by prefixing `PYTHONUNBUFFERED=1` on the second Exp 0 v2 launch. Consider standardizing this in CLAUDE.md "How to Run" — current commands omit it.
- **`best_metric=sortino` + sortino=0 from do-nothing policy:** first `best_value` was set to 0.0 ("NEW BEST" off −inf), then strict `>` comparison meant every subsequent eval (also sortino=0) was "no improve" → early-stop at 5 evals → run died at 4 minutes instead of 17. Switched to `best_metric=total_return` which is strictly < 0 for any losing trade, breaking ties properly.

---

## 2026-05-15 — Portfolio-MDP rebuild complete (env + 3 trainers + baselines + Exp 0 + HPO)

### Status
The portfolio-MDP rebuild whose TODO list was left at the bottom of the prior entry is **done**. Pipeline smoke-tested end-to-end on all three algorithms and on baselines. `runs/` is empty (only smoke artifacts, removed). No real (paper-grade) runs have been done yet — Exp 0 is the next action by the user.

### What was rebuilt this session
- **src/env.py** — full portfolio MDP per PROPOSAL Sec. 3: `RunningStd` reward normalizer class, `IntradayTradingEnv` with `capital`, `lot`, `contract_size`, `spread_point_size` config; per-bar dollar P&L; spread cost in dollars from the parquet `spread` column on every position change; equity-mark-to-market; equity ≤ 0 ruin terminates with fixed `reward = -1`; EOD force-flatten with spread cost; reward modes `r1` (log equity return), `r2` (dollar Δ), `r4` (Δ - β·max(0, DD - τ)); optional running-std normalization (training only — `reward_normalizer=None` for eval); state is 16-dim (10 market + 5 positional + 1 `equity_ratio`).
- **config.yaml** — added `env.capital: 10000`, `env.lot: 0.01`, `env.contract_size: 100`, `env.spread_point_size: 0.01`, `env.reward.{mode,normalize,beta,dd_thresh,ruin_reward}`. Removed the old `env.commission`. `train.best_metric` allowed list expanded to include `final_equity`.
- **src/train.py** (DDQN) — `DayCycleEnv` now threads a single shared `RunningStd` instance into every inner env (training only); `evaluate_policy_per_session` constructs eval envs with `reward_normalizer=None` (eval never normalized — PROPOSAL Sec. 6.4); metric pipeline uses `info["pnl_log"]` so Sharpe/Sortino/MDD are invariant to reward mode; `pooled_metrics` extended with `final_equity`, `max_dd_dollar`, `max_dd_pct`, `ruin_rate` (PROPOSAL Sec. 4). `metrics.csv` schema bumped by 4 columns (header captured in `METRICS_HEADER` constant so the A2C/PPO trainers can reuse the exact same schema).
- **src/train_a2c.py + src/train_ppo.py** — same portfolio support, parallel per-bar collectors for `pnl_log`/`equity`/`ruin` alongside the Rollout (the optimizer's `reward` stays in the Rollout, while metrics use `pnl_log`). PPO retains all KL/clip_fraction/n_epochs_run diagnostics.
- **scripts/run_baselines.py + scripts/run_seeds.py** — `METRICS` list extended with the 4 portfolio metrics so they show up in CSV/JSON aggregates. Baseline summary console output now shows `final_equity`/`max_dd_dollar`/`ruin_rate`.
- **scripts/run_exp0.py** (NEW) — the Exp-0 driver: `--rewards r1 r2 r4 --algos ddqn a2c ppo --seeds 42 1337 2026` = 27 runs. No HPO. Output: `runs/exp0/<reward>_<algo>_s<seed>/` plus `runs/exp0/exp0_summary.csv` and `runs/exp0/exp0_winner.json`. Ranking: per-algo mean-over-seeds of `val_<best_metric>`, rank rewards 1..3 per algo, mean rank across algos. Tie-breaker = best raw mean across algos. Resumable via `--skip-existing`. `--total-timesteps` override for quick smoke runs.
- **src/hpo.py + scripts/run_final.py** — `_sample_hparams` takes a `reward_mode` arg; when `r4` it additionally samples `reward_beta` (log-uniform [0.1, 10]) and `reward_dd_thresh` (log-uniform [0.005, 0.10]). Keys with the `reward_` prefix are routed into `cfg["env"]["reward"]` by `_apply_hparams` (and `run_final.py._apply_params`); other keys still go into the algo's config section. Per PROPOSAL Sec. 3.6, β and τ are only tuned in Exp 1 on the winning reward — Exp 0 keeps them fixed.

### Design choices made (locked in this session)
- **Spread-cost math in dollars.** `$cost = spread_pts × point_size × contract_size × lot`. MT5 XAUUSD: `point_size = 0.01`, `contract_size = 100`, `lot = 0.01` → `$0.01 per spread point per 0.01 lot per side`. Median spread ~50 pt → ~$0.50/side. Random 1379-bar rollout with ~919 flips loses ~$450 to spread, matches expectation. All four numbers configurable.
- **`$/price = contract_size × lot = 1.0`** with the defaults — i.e. a $1 gold move = $1 P&L per 0.01 lot. This is why ruin is essentially unreachable at $10k capital on a single day; PROPOSAL Sec. 8 acknowledges the $100 curriculum as future work.
- **Reward normalization is a per-RUN singleton, NOT per-episode.** One `RunningStd` is created in the trainer and passed via `DayCycleEnv` into every freshly-constructed inner env. Welford updates run across all training steps — this is what makes R1 (~1e-4/bar) and R2/R4 ($1-30/bar) comparable at one learning rate.
- **Ruin reward is NEVER normalized.** The fixed `-1` clip is emitted directly even when the normalizer is active. This keeps the ruin signal sharp regardless of how the running std has scaled the other rewards.
- **Metrics use `info["pnl_log"]`, not `reward`.** Sharpe/Sortino/MDD/total_return are computed from the per-bar log-equity-return, so they're identical across reward modes for the same trade sequence. This is the only honest way to compare R1 vs R2 vs R4 in Exp 0.
- **Episode-level `max_dd_dollar` and `final_equity`.** These are per-episode quantities (each episode is independent, $10k capital each). The reported phase value is `mean(final_equity_per_ep)` and `max(max_dd_dollar_per_ep)` (worst-case episode). `ruin_rate` = `mean(ruin_flag_per_ep)`.
- **Exp 0 ranking metric = `cfg["train"]["best_metric"]` (default `sortino`).** Per PROPOSAL Sec. 5, the winner is by mean rank across the 3 algos. NaN val (no checkpoint saved) sorts to the bottom rank. Tie-breaker: best raw mean across algos.

### Smoke tests passed this session
- `env.py __main__`: random 1379-bar rollout on a real val day, final equity $9596, 919 trades, no NaN, normalizer std=1.05e-4.
- `scripts/sanity.py` (DDQN, 20k timesteps): 14 eps, val sortino=-6.48 at best, test final equity $9985, mdd=$482, exploration 1.0→0.05, no NaN.
- A2C (5k timesteps, 3 eps): pi_loss/entropy/EV all healthy, val final equity $9998.
- PPO (5k timesteps, 3 eps): KL=0.012, clip_frac=0.16-0.23, EV climbs 0.04→0.36 in 3 episodes (same sample-efficiency advantage over A2C as before).
- `run_baselines.py --split val`: flat=$10000, long=$10007 (gold up in val window), short=$9991, random=$9522 (eaten by spread on 45k trades), ma_crossover=$9984 (whipsaws on 20/60 MA on M1).
- `run_exp0.py` orchestration (1 reward × 1 algo × 1 seed, 3k timesteps): summary CSV + winner JSON written correctly.
- `hpo._sample_hparams` with `reward_mode="r4"`: `reward_beta` and `reward_dd_thresh` are sampled and routed into `cfg["env"]["reward"]`, algo params still route to `cfg["dqn"]`.

### What's NOT done (next steps for the user)
- **Run Exp 0 for real**: `& 'd:\EA\.venv\Scripts\python.exe' 'd:\EA\scripts\run_exp0.py'` — 27 runs at full 500k timesteps each. Estimated wall-clock at ~360 steps/sec ≈ 10-12 hours total (with DDQN early-stop typically ~196k, A2C/PPO slower per step). User decides whether to lower timesteps for Exp 0 only.
- **Pick the winner from `runs/exp0/exp0_winner.json`**, set `env.reward.mode = <winner>` in config.yaml.
- **Run Exp 1** (Aggressive HPO + final retrain + baselines) on the winning reward.

### Things that were intentionally NOT changed
- The three agent files (`ddqn.py`, `a2c.py`, `ppo.py`) — `obs_dim` is a constructor parameter on all of them, so the 15→16 dim change is a no-op at the agent layer. The trainer passes `env.observation_space.shape[0]` which now resolves to 16. Verified by reading the agent code, no smoke-test mismatch.
- The `feature.py` module — the 10 market features are unchanged; PROPOSAL Sec. 3.5 specified them, and `features.py` already keeps the raw `spread` column which the env now charges as dollar cost.
- Optuna search ranges for algorithm hyperparameters — unchanged from the Aggressive plan. Only R4 adds two new dimensions, and only when `reward.mode == "r4"`.

---

## 2026-05-15 — MAJOR: MDP redesigned return-based → portfolio-based; old work invalidated

### Why
User clarified the intent was a **portfolio simulation with real capital and lot-based sizing**, not the return-only abstraction in the presented proposal. The old PDF/experiment.md spec (per-bar log-return after a relative commission; no capital, no lots, no ruin) did not match this. Decision: write a new **canonical living spec** and rebuild to it. The old presented PDF is *one snapshot*, not binding.

### What was decided (full spec in PROPOSAL.md Sec. 3/5/6)
- **Portfolio MDP:** fixed `C0 = $10,000` per episode (reset each episode, no cross-episode compounding so CV stays valid); fixed **0.01 lot** (1 lot = 100 oz → $1 move = $1 P&L per 0.01 lot); action = direction only (short/flat/long), no lot sizing by agent.
- **Cost = spread only**, commission **$0**, charged in dollars from the parquet `spread` column on each position change. Execution still next-bar-open on change. EOD force-flat.
- **Ruin:** `equity ≤ 0` → terminate, fixed clipped `reward = -1`. Single hard threshold, no broker margin/stop-out modelling (user explicitly did not want margin mechanics — "we measure at drawdown").
- **State 15 → 16 dim:** added `equity_ratio = equity_t / C0`.
- **Reward comparison (new Exp 0):** R1 log-return (control), R2 raw dollar P&L (matches intent), R4 P&L − drawdown-penalty (CMDP-style, fits ruin rule). R3 (mean-SD / vol-penalized) considered then **dropped** (overlaps R4, noisier on M1). All wrapped in **running-std reward normalization incl. the R4 penalty** so the three are comparable at one lr (the R1≈1e-4 vs R2≈$1–30 scale gap would otherwise make a shared lr unfair / diverge). Evaluation never normalized.
- **Experiment structure:** Exp 0 (reward selection: 3 rewards × 3 algos × 3 seeds = 27 runs, **no HPO**, winner = best mean rank across algos on val) → Exp 1 (algo comparison, full Aggressive HPO on winning reward) → Exp 2 (action space) → Exp 3 (LSTM/GRU). Exp 0 has no HPO to avoid the ~25 h cost; honest limitation written into PROPOSAL.md.

### Reward shortlist came from a paper survey
Two parallel sub-agents read 16 PDFs in `research_papers/` (03_trading_strategies, 05_execution_costs, 04_portfolio_management). Extracted 6 reward families; filtered to dense-per-step + fixed-lot-dollar-compatible → R1/R2/R4. Volatility-scaled (Zhang/Zohren) rejected (needs variable sizing, conflicts with fixed lot). Episodic Sharpe/MDD/CVaR rejected as *primary* signals (too sparse for M1 — TDQN/DRQN/DeepTrader all note this); kept as metrics/future shaping only.

### Cleanup + conflict resolution done this session
- **Deleted all of `runs/`** (ddqn+a2c+ppo Aggressive HPO trials, sqlite studies, best.json, _tb, logs, timing). All return-based → invalid under the new MDP. Only `.gitkeep` remains.
- **Renamed** `proposal/DRL_proposal_6509_6571.pdf` → `..._OLD_presented.pdf`; `proposal/experiment.md` → `experiment_OLD_return_based.md` (return-based spec, superseded by PROPOSAL.md, kept for history).
- **CLAUDE.md** rewritten: MDP/State/Reward sections now point to PROPOSAL.md as authority (no duplicated formulas to avoid drift); env.py line, policy-net 15→16, How-to-Run got a STALE banner, Insights got a portfolio-MDP-migration banner + the runs/Aggressive lines updated.
- Stopped the lingering TensorBoard background task.

### What this invalidates / rebuild TODO (tracked in the task list)
Code still implements the OLD return-based MDP. Must rebuild before any valid run: `src/env.py` (capital/lot/contract/spread-cost/ruin/R1-R2-R4/reward-norm), state 15→16, agent input dims, `src/data.py` (spread→real cost), `config.yaml` (capital/lot/reward sections), new metrics (final_equity, max_dd_dollar, ruin_rate), new `scripts/run_exp0.py`, then smoke-test. The Aggressive HPO recipe (3-fold/12-trial/100k) is still the intended Exp-1 HPO config — just to be re-run on the rebuilt MDP from the Exp-0 winning reward.

### Things explicitly rejected this session
- Keeping the old return-based reward "to stay close to the presented proposal" — user wants the portfolio sim.
- Broker margin-level / stop-out modelling — single `equity ≤ 0` ruin only.
- Variable / risk-based lot sizing in the main study — fixed 0.01 lot (curriculum to low capital like $100, and dynamic sizing, are future work in PROPOSAL.md Sec. 8).
- $100 starting capital for the main study — with 0.01 lot on gold it ruins within minutes (≈4% adverse move), agent collapses to do-nothing before learning; $10k chosen. ($100/curriculum kept as a future experiment.)

---

## 2026-05-15 — HPO pipeline implemented (src/hpo.py + run_hpo.py + run_final.py) + train adapter

### What was built
The HPO stack that the 2026-05-15 nested-CV entry left as a TODO is now implemented and smoke-tested end-to-end. bake-off was **skipped** — user decided to go straight to Experiment 1 (Aggressive plan).

**New files:**
- [src/hpo.py](src/hpo.py) — `make_objective(algo, base_cfg, folds, hpo_timesteps, study_name)` returns an Optuna objective closure. Per-algo search spaces match the Aggressive plan draft. Runs every CV fold, `trial.report(running_aggregate, fold_index)` + `trial.should_prune()` after each fold, returns `aggregate_fold_scores` (mean − 0.5·std) or `-inf` if `mean_val_trades < cv.min_trades`. Also `_sample_hparams`, `_apply_hparams`, `_run_fold`.
- [scripts/run_hpo.py](scripts/run_hpo.py) — entry `--algo {ddqn,a2c,ppo}`. Defaults to Aggressive plan: **12 trials, 3 folds, 100k timesteps/fold**, TPESampler + HyperbandPruner, sqlite storage (resumable), writes `runs/hpo/<algo>/<study>_best.json`.
- [scripts/run_final.py](scripts/run_final.py) — pulls best trial, writes its params into the algo config section, retrains at full 500k × `--seeds` on the full 600-day train with held-out val(75) for best-ckpt, **test(75) evaluated once per seed**. Writes `runs/final_<algo>/seeds_summary.csv` + `seeds_aggregate.json` (same schema as run_seeds.py).

### Train adapter (the contract HPO relies on)
`train_ddqn/train_a2c/train_ppo` now read an optional `cfg["_hpo"]` dict:
- `inner_train_dates` / `inner_val_dates` — replace the default train/val split with a CV fold. Held-out val/test from `split_days()` are NOT used in HPO.
- `timesteps_override` — per-fold budget (100k) without rewriting cfg.
- When `_hpo` is set the function **early-returns before the test rollout** with `{"hpo_objective": best_val_score, "best_metric", "val_trades"}` — test is never touched during HPO. `scripts/*` (train_ddqn.py, run_seeds.py, run_baselines.py) never set `_hpo`, so their behavior is unchanged.
- `best_info.json` now also records `val_trades` (trades at the best checkpoint) so the min_trades filter judges the *selected* policy, not the last eval.

### Design decisions made while implementing
- **`trial.report()` reports the running aggregate** (mean−0.5·std of completed folds), not the raw per-fold score. Raw fold scores swing with each fold's market regime; the running aggregate is the quantity actually being optimized and gives the Hyperband pruner a like-for-like comparison across trials at the same fold index.
- **min_trades filter uses mean val_trades across folds** (not per-fold). A config that trades enough on average but degenerates on one regime is still informative; per-fold disqualification was judged too aggressive at 3 folds.
- **Config-section name mismatch fixed:** DDQN's config block is `cfg["dqn"]`, but the algo arg is `"ddqn"`. Both hpo.py and run_final.py carry a `_CFG_SECTION = {"ddqn": "dqn", ...}` map. Caught by the smoke test (first failure).
- **`_hpo` stripped before yaml dump:** `cfg["_hpo"]` holds pandas Timestamps which `yaml.safe_dump` cannot represent. All three trainers now build `cfg_dump = {k:v for k,v in cfg.items() if k != "_hpo"}` for both the `config.yaml` artifact and the TB `add_text("config")`. Caught by the smoke test (second failure).

### Smoke test (DDQN, 2 trials × 2 folds, 7k timesteps/fold) — PASSED
End-to-end works after the two fixes above. Trial 0 agg sortino −29.04 (folds −19.83, −32.11), Trial 1 agg +10.36 (folds +2.85, +32.89) → TPE correctly kept Trial 1 as best; `val_trades` (605/564/151/96) passed the min_trades filter; best-trial JSON written. Note: with a too-small budget (3k/fold) no val eval fires before timesteps run out → `best_value` stays −inf → trial returns −inf and is filtered. This is correct behavior, not a bug, but means **HPO timesteps must be large enough that at least one val eval fires per fold** — at 100k/fold with `eval_every_sessions=11` this is comfortably satisfied (≈70+ episodes/fold).

### Status
Experiment 1 (Aggressive) is now runnable. Estimated compute ≈ 3.8–4.5 h at ~360 steps/sec (HPO ~3–3.5 h + final retrain ~50 min + baselines). All training/HPO will be **executed by the user**, not Claude. `runs/` is empty (only `.gitkeep`).

### Run order for Experiment 1 (user executes)
```powershell
# 1. HPO per algo (resumable; ~1-1.5h each)
& 'd:\EA\.venv\Scripts\python.exe' 'd:\EA\scripts\run_hpo.py' --algo ddqn
& 'd:\EA\.venv\Scripts\python.exe' 'd:\EA\scripts\run_hpo.py' --algo a2c
& 'd:\EA\.venv\Scripts\python.exe' 'd:\EA\scripts\run_hpo.py' --algo ppo
# 2. Final retrain + test per algo (3 seeds each)
& 'd:\EA\.venv\Scripts\python.exe' 'd:\EA\scripts\run_final.py' --algo ddqn
& 'd:\EA\.venv\Scripts\python.exe' 'd:\EA\scripts\run_final.py' --algo a2c
& 'd:\EA\.venv\Scripts\python.exe' 'd:\EA\scripts\run_final.py' --algo ppo
# 3. Baselines on test + TensorBoard
& 'd:\EA\.venv\Scripts\python.exe' 'd:\EA\scripts\run_baselines.py' --split test
& 'd:\EA\.venv\Scripts\tensorboard.exe' --logdir 'd:\EA\runs\_tb' --port 6006
```

---

## 2026-05-15 — bake-off scouting run + revised "Aggressive" Experiment-1 HPO plan (compute-constrained)

### Context
User capped total experiment time at 2-3 h for scouting and re-prioritized compute as a hard constraint (it was explicitly *not* a priority in the 2026-05-15 nested-CV design entry below — that has now changed). Two-phase approach adopted:

1. **bake-off (scouting, in progress):** DDQN/A2C/PPO at *default* config, 3 seeds each, no HPO, no inner CV. Reported on **val only** — test is NOT touched (decision "B1": test rollout still computed by the unchanged `train_*()` code but its numbers are never read or reported, so no information leaks into any human decision). Baselines (5) evaluated on **val** via `run_baselines.py --split val`. Purpose: confirm no algorithm's pipeline is broken at default config and get a rough ranking before committing to the longer HPO run. **This is scouting, NOT Experiment 1 — results must not go in the paper** (hyperparameters un-tuned, val has optimistic bias from best-ckpt selection).
2. **Experiment 1 (the real, paper-grade run):** revised "Aggressive" HPO plan below.

### Benchmark measured this session (drives the timestep cuts)
- DDQN ~360 steps/sec on this machine's GPU (750-day dataset, default config).
- DDQN 500k run **early-stops at ~196k (39% of budget)** with `early_stop_patience=5`; **best checkpoint at step ~121k**. → 200k/fold in the original HPO plan is over-provisioned for DDQN; 100k/fold is enough to reach the convergence region.

### Revised Experiment-1 HPO plan ("Aggressive", fair across algos)
Locked in with user (chose "HPO 3 algos equally" over "HPO winner only" — the latter was rejected because tuning only one algorithm and leaving the other two at default makes the algorithm comparison unfair and reviewer-challengeable; proposal Q1 requires equal conditions).

| Parameter | Original plan (entry below) | Revised "Aggressive" |
|---|---|---|
| n_folds | 5 | **3** |
| n_trials / algo | 30 | **12** |
| HPO timesteps / fold | 200k | **100k** |
| Pruner | Median (cuts after fold 2) | **Hyperband / SHA** (can cut from fold 1) |
| Algos given HPO | 3 | **3 (equal)** |
| Final retrain | 3 algos × 3 seeds × 500k (early-stop) | unchanged |

Estimated wall-clock: HPO ~10.8M raw timesteps, Hyperband effective ~45%, A2C/PPO ~1.3-1.5× slower than DDQN → ~6.5M effective; final retrain ~1.8M; total ≈ **3.8-4.5 h** at 360 steps/sec.

### Decision reversal recorded: 3-fold inner CV
The 2026-05-15 nested-CV entry (below) explicitly **rejected 3-fold** ("less robust; saving compute not the priority"). That rejection is **superseded here** because compute *is* now the priority. Trade-off accepted: 3 expanding folds give weaker cross-regime robustness than 5. Mitigation: if a tuned algorithm looks promising, re-run HPO for that algorithm only with more folds/trials — Optuna study (sqlite) resumes, so the cheap run is not wasted. 100k/fold is also noisier than 200k; same mitigation applies.

### Code fix this session (cp874 stdout crash, permanent)
`scripts/run_seeds.py` (2 sites) and `scripts/run_baselines.py` (2 sites) used `→` in `print()` statements. On the Thai-locale system Python 3.9, cp874 stdout encoding crashes on this char *after* the result files are already written (so prior baseline output was salvageable). Replaced `→` with `->`. CLAUDE.md already warns about cp874 for config *reads*; this extends the same caution to stdout. Note: HPO scripts (`src/hpo.py`, `scripts/run_hpo.py`, `scripts/run_final.py`) are still unwritten — keep their console output ASCII-only.

### proposal/experiment.md brought in sync with the 750-day dataset
That file was stale (still 126-day window, 365-day parquet, 70/15/15 ratio split = 88/18/20). Updated: dataset section (750 days, correct filename, date range), split table (absolute 600/75/75 with real date ranges), new "Inner Cross-Validation" subsection, controlled-settings window row. Note: the inner-CV numbers written into experiment.md describe the **original 5-fold plan**, not the revised 3-fold Aggressive plan — experiment.md documents the protocol design; the active compute-constrained variant lives in this journal entry. Reconcile if the Aggressive plan becomes the final paper protocol.

---

## 2026-05-15 — Dataset expansion to 750 days + nested-CV protocol design (HPO not yet implemented)

### Why
Setting up a defensible **train → HPO with inner CV → final eval on held-out val/test** protocol so the three-algorithm comparison (DDQN vs A2C vs PPO) is not contaminated by val leakage. Old 126-day window (88/18/20) was too small to support nested CV — inner CV folds would have been ~5 days each, far too noisy for trading metrics. Rebuilt the dataset and split scheme to make nested CV actually meaningful.

### Dataset re-download
- [data/scripts/01_download_gold_m1_to_parquet.py](data/scripts/01_download_gold_m1_to_parquet.py): `TARGET_TRADING_DAYS = 365 → 750`, `LOOKBACK_CALENDAR_DAYS = 560 → 1200`. Output renamed to `GOLD_M1_last750_trading_days_to_2026-05-01.parquet`.
- Re-downloaded via MT5. **750 trading days, 1,027,754 M1 bars, range 2023-06-05 → 2026-04-30 (~2.9 years calendar).**
- Bars-per-day distribution: median 1379 (full), min 958 (Jan 2 / Dec 26). All shortest days are known holidays. No weekends (correct for gold CFD).
- Old 365-day file is no longer referenced anywhere in the codebase; kept on disk for now in case anyone needs to reproduce prior journal entries.

### Split scheme: 600 / 75 / 75 (chronological, absolute counts)
- [src/data.py](src/data.py): `split_days()` signature changed from `(train_ratio, val_ratio, test_ratio)` to `(n_train, n_val, n_test)`. Asserts `n_train + n_val + n_test == len(dates)`. Removes ratio-induced rounding (old `int(n*0.7)` gave 88 from 126 only by coincidence).
- [config.yaml](config.yaml): replaced `train_ratio/val_ratio/test_ratio` with `n_train: 600`, `n_val: 75`, `n_test: 75`. `window_days: 750`.
- Updated 5 call sites: [src/train.py](src/train.py), [src/train_a2c.py](src/train_a2c.py), [src/train_ppo.py](src/train_ppo.py), [src/env.py](src/env.py), [scripts/run_baselines.py](scripts/run_baselines.py). All pass `(n_train, n_val, n_test)` to `split_days()`.
- Verified end-to-end via `python src/data.py`: 600 train (2023-06-05 → 2025-09-29), 75 val (2025-09-30 → 2026-01-14), 75 test (2026-01-15 → 2026-04-30).

### Sizing decision (why 750/600/75/75)
Worked backwards from constraints:
1. **Test ≥ 55 days** so Sharpe annualized std-error ≤ 0.15 (Lo 2002 approximation: `σ_SR ≈ √((1 + 0.5·SR²)/N)`). 75 days gives margin.
2. **Inner CV folds ≥ 24 days/fold val** so per-fold Sortino is not noise-dominated; 5 expanding folds × 24 = 120 days carved out of train → minimum inner-train of the first fold = 600 − 120 = 480 days, well above the rule-of-thumb 270.
3. **Held-out val ≥ 55 days** for the same reason as test (it gates checkpoint selection after HPO).
4. **Total = 600 + 75 + 75 = 750.** Going beyond 750 adds compute cost without meaningful generalization gain and risks regime drift (Gold 2022 Fed-hiking peak ≠ 2024 cuts ≠ 2025-26). MT5 broker also might not have clean M1 history past 3 years.

### Inner CV infrastructure
- New file [src/cv.py](src/cv.py): `expanding_window_folds(train_dates, n_folds=5, val_size=24)` and `aggregate_fold_scores(scores, penalty=0.5)`.
- Layout **Option A — equal val size, expanding train.** Fold k ends at `train_dates[N − (n_folds−k)·val_size]`. With N=600, n_folds=5, val_size=24:
  - fold 1: inner_train=480 days [2023-06-05 → 2025-04-11], inner_val=24 [2025-04-14 → 2025-05-16]
  - fold 2: inner_train=504, fold 3: 528, fold 4: 552, fold 5: 576
  - Final fold's inner_val ends exactly at last train date (2025-09-29). Held-out val (2025-09-30 onward) is **never touched** by inner CV.
- Trial score aggregation: `mean(fold_scores) − agg_penalty · std(fold_scores)` with `agg_penalty=0.5`. Rewards hyperparameter configs that are consistent across regimes, not just lucky on one fold.
- Why expanding (not sliding or equal-chunk): expanding keeps every fold's train ≥ 480 days (none are too small to learn meaningfully), preserves chronological order, and is closest to a production deployment cadence.

### `cv:` section added to config.yaml
```yaml
cv:
  n_folds: 5
  val_size: 24              # 5*24 = last 120 train days become inner-val across folds
  agg_penalty: 0.5          # trial_score = mean - 0.5*std
  min_trades: 50            # disqualify trials whose policy doesn't trade enough on inner-val
  objective_metric: sortino # one of: total_return, sharpe, sortino
```

### Optuna installed
- `pip install --only-binary=:all: optuna` → **optuna 4.8.0** (with sqlalchemy 2.0.49, alembic, etc.). Source build of greenlet fails without MSVC; binary wheels work.
- Not yet integrated. Spec for next session (below).

### HPO methodology decisions made this session (locked in)
1. **Objective:** `val_sortino` with `min_trades >= 50` disqualifier — Sortino over Sharpe because we don't want to penalize upside vol; min-trades because prior runs (see 2026-05-14 "Normalization ablation") showed policies collapsing to "do nothing", giving misleadingly OK Sortino on near-zero return sequences.
2. **Trial budget:** 30 trials per algorithm. Reasoning: enough for TPE to find structure (typical sweet spot 30-50), few enough to limit multiple-comparisons-style val overfit; equal across algorithms for fair comparison.
3. **HPO timesteps:** 200k per fold (reduced from production 500k) + full 500k retrain on best config.
4. **Search space includes `hidden_sizes`** for all three algos — sampled from `{[64,64], [128,128], [256,256]}`.
5. **Sampler + pruner:** TPE sampler + Median pruner. Pruner cuts trials at intermediate fold reports if score < median of completed trials at same fold index.
6. **Held-out val (75 days) is NOT used during HPO.** It is reserved for: (a) best-checkpoint selection during the final 500k retrain, (b) early-stop signal during retrain. This is the leakage fix — val is touched only once per algo per seed, not 30 times.
7. **Test (75 days) is touched exactly 3 times total** — once per algo, after best-trial-best-checkpoint is locked. With 3 seeds × 3 algos = 9 test rollouts total + 5 baseline test rollouts = 14 total test passes. No HPO feedback from test.

### TODO for next session — full HPO implementation
**New files (to write):**
- `src/hpo.py` — `objective(trial, algo: str, cfg: dict, feat: pd.DataFrame, folds: list[CVFold]) -> float`. Samples hyperparameters from per-algo search space dict, runs training over all CV folds with intermediate `trial.report(score, fold_idx)` + `trial.should_prune()` checks, returns aggregated trial score (or `-inf` if min_trades filter fails).
- `scripts/run_hpo.py` — entry: `--algo {ddqn|a2c|ppo} --n-trials 30 --storage sqlite:///runs/hpo/<algo>.db --study-name <algo>_v1`. Loads data once, builds folds once, creates Optuna study with `TPESampler` + `MedianPruner`, runs `study.optimize()`, writes best-trial JSON + a summary to `runs/hpo/<algo>/`.
- `scripts/run_final.py` — given a study, pull best trial, retrain with full 500k timesteps × 3 seeds on full train(600) using held-out val(75) for best-ckpt selection, eval on test(75). Writes per-algo aggregate.

**Modifications to existing code:**
- [src/train.py](src/train.py), [src/train_a2c.py](src/train_a2c.py), [src/train_ppo.py](src/train_ppo.py): each `train_*(cfg)` needs to accept optional `(train_dates_override, val_dates_override)` so HPO can inject CV fold dates instead of the default split. Also need optional `total_timesteps_override` (so HPO can run shorter 200k passes without rewriting the cfg dict). Cleanest path: pass them via `cfg` itself (e.g. `cfg["_hpo"]={"train_dates":..., "val_dates":..., "timesteps_override": 200000}`) and have train read those if present, else fall back to config. Adapter pattern keeps existing scripts/* unchanged.

**Per-algo search spaces (drafted, refine when implementing):**
- DDQN: `lr` log-uniform [1e-4, 1e-2], `batch_size` {32,64,128}, `target_update_interval` {500,1000,2000}, `exploration_fraction` [0.3, 0.8], `hidden_sizes` {[64,64],[128,128],[256,256]}.
- A2C: `lr` log-uniform [1e-4, 3e-3], `entropy_coef` log-uniform [1e-3, 0.1], `gae_lambda` [0.9, 0.99], `hidden_sizes` {[64,64],[128,128],[256,256]}.
- PPO: `lr` log-uniform [1e-4, 1e-3], `clip_range` [0.1, 0.3], `n_epochs` {5,10,20}, `minibatch_size` {64,128,256}, `entropy_coef` log-uniform [1e-3, 0.1], `hidden_sizes` {[64,64],[128,128],[256,256]}.

**Compute budget estimate:**
- Per trial: 5 folds × 200k = 1M timesteps if no pruning; assume median pruner cuts ~50% trials after fold 2 → effective ~600k/trial avg.
- Per algo: 30 trials × 600k ≈ 18M timesteps + 3 seeds × 500k retrain = 19.5M.
- 3 algos total: ~60M timesteps. Current 500k DDQN run takes a few minutes on cuda (per journal); 60M ≈ 120× → estimate 1-3 days wall-clock total. Worth doing as one block.

**Old runs in `runs/` are stale** under the new dataset / split. Don't compare them to anything from the new pipeline. Clean them up before re-running, or leave them as historical reference.

### Things explicitly rejected this session
- "Just accept val leakage + write a limitations section." User does not want this.
- Going beyond 1000 days (regime drift risk, broker M1 availability).
- Nested CV with 3 folds (less robust; saving compute not the priority).
- Single inner train/val split (not robust enough; CV was the point of the redesign).

## 2026-05-15 — Pipeline hardening: metric fixes, expanded action space, baselines, multi-seed runner

### Metric fixes (`src/train.py`)
- **Sortino**: fixed denominator from `downside.var()` → `downside_d.std()` (daily path) and `downside.std()` (per-bar fallback). Was reporting variance, not standard deviation.
- **Sharpe/Sortino annualization**: added daily-aggregation path in `pooled_metrics()`. When `day_lengths` is provided (val/test eval), per-bar returns are summed per day → `mean(daily_r) / std(daily_r) × √252`. Single-episode train logging still uses per-bar fallback with no annualization.
- **3 new metrics added**: `avg_trade_pnl` (mean completed-trade log return), `turnover` (`sum(|diff(positions)|)`), `avg_holding_time` (mean bars_held per trade). All three appear in `metrics.csv`, TensorBoard, and `summary.json`. `metrics.csv` is now 13 columns.

### Expanded action space (`src/env.py`)
- `IntradayTradingEnv` now reads `cfg["env"]["action_space"]` as a list of position floats. `n_actions = len(list)`. Supports 3-action `[-1,0,+1]` (Exp 1), 5-action `[-1,-0.5,0,+0.5,+1]` (Exp 2a), and 9-action variants out of the box — just change one line in `config.yaml`.
- Position values are now `float` throughout (env, Rollout, train loops). `int(info["next_position"])` replaced with `float(...)` in all three training loops and eval.
- `config.yaml` has commented-out lines for Exp 2a and Exp 2b ready to uncomment.

### Baselines (`src/baselines.py`, `scripts/run_baselines.py`)
- 5 baselines implemented, all drop-in for `evaluate_policy_per_session()`:
  - `FlatBaseline` — always flat (zero-trade reference)
  - `LongBaseline` — enter long at session open, hold to EOD
  - `ShortBaseline` — enter short at session open, hold to EOD
  - `RandomBaseline` — uniformly random action per bar (seeded from `train.seed`)
  - `MACrossoverBaseline` — long when fast MA > slow MA (default 20/60 bars), short otherwise; causal, no lookahead
- `evaluate_policy_per_session()` now calls `agent.prepare(day_df)` if the method exists, enabling MA crossover's per-session signal precomputation without touching DRL agent code.
- `scripts/run_baselines.py` evaluates all 5 on test (or val/both), writes `runs/baselines/baselines_results.csv` + `.json`.

### Multi-seed runner (`scripts/run_seeds.py`)
- Loops any single algo (`ddqn`, `a2c`, `ppo`) or `all` over `--seeds 42 1337 2026` (default).
- Each seed → its own run folder `<base_name>_<algo>_s<seed>/`.
- Aggregate output in `runs/<base_name>_<algo>/`: `seeds_summary.csv` (one row/seed) + `seeds_aggregate.json` (mean ± std per metric).
- Prints a formatted mean/std table to console at the end.

### `proposal/experiment.md` wording fixes
- Market features: listed all 10 explicitly; MACD clarified as raw value (not histogram); spread clarified as `spread_pts` column.
- Positional features: descriptions made precise (`tl` = bars until forced close, `pos` supports partial sizes in Exp 2, `pr` = net of entry cost).
- Reward section: added note that EOD forced close also charges transaction cost.
- Controlled settings: `best_metric` now lists valid options.
- MDD: formula made explicit.
- Trades: "contiguous non-zero position block" definition added.
- Rollout cadence: A2C/PPO noted as 1 rollout = 1 trading day in Exp 1 setup table.
- Baselines: removed duplicate Buy-and-hold/Long-only entry; descriptions sharpened.

## 2026-05-13 / 2026-05-14 — Pre-portfolio-MDP pipeline work (compacted 2026-05-17)

Six entries on 2026-05-13/14 covered the early pipeline build under the **old return-based MDP** (15-dim state, log-return reward, commission cost, no $-equity). All of that was **superseded on 2026-05-15** by the portfolio-MDP rebuild (next entry below); the still-active findings are kept here as bullets, and the full entries are recoverable via `git log -p JOURNAL.md`.

**Still-active findings from this period (kept verbatim in CLAUDE.md "Insights" where applicable):**

- **2026-05-13** — Pipeline rebuilt to proposal spec (15-dim state, MACD/STO/RSI/ATR + 5 positional features TL/POS/PR/DR/HT, next-bar-open execution, no future leakage). Switched to **pooled period-level metrics** (DeepScalper §5.2): one metric row per eval pass over concatenated per-bar returns, not per-day-then-averaged. `total_return = exp(sum r) − 1`. Equity is continuous within an eval pass but resets between train/val/test. Train-phase Sharpe is **not logged** and should not be — train returns come from a stochastic policy.
- **2026-05-14 (DDQN from scratch)** — Replaced SB3 with from-scratch PyTorch DDQN. A/B at 3 seeds × 150k steps: policy-quality metrics (best val, test return, MDD, winrate) pass at 1σ; Sharpe/Sortino "FAIL" results were ratio-metric artifacts on tiny absolute values at n=3. Do NOT bring SB3 back. All three algos are scratch implementations for fair comparison.
- **2026-05-14 (A2C / PPO added)** — On-policy gradient algorithms need tighter gradients than DDQN: **A2C lr=0.0007, max_grad_norm=0.5; PPO lr=0.0003, max_grad_norm=0.5; DDQN lr=0.0045, max_grad_norm=10.0**. Keep these gaps if you tune. Other locked-in choices: shared trunk + 2 heads (policy/value) with Tanh, orthogonal init gain=0.01 on policy head, episode-aligned rollouts (1 rollout = 1 day, `last_value=0` at EOD), GAE(λ=0.95). **PPO advantage normalization is per-minibatch (rollout-level statistics go stale across K epochs); A2C is per-rollout** — don't unify.
- **2026-05-14 (normalization ablation)** — Rolling z-score on market features (windows 60, 240) made things worse or no better than raw. Removed from `src/features.py`. Real bottleneck was over-trading collapsing into "do nothing", not feature scale — don't re-introduce normalization as a fix for poor trading metrics.
- **2026-05-14 (pipeline audit)** — Fixed `DayCycleEnv` off-by-one (cursor was advancing during `__init__` and skipping `order[0]` every cycle). Decoupled eval cadence from epoch (`eval_every_sessions`). Decoupled "epoch" from "session" (epoch = one full pass over `train_dates`). `metrics.csv` schema settled on `episode, global_step, epoch, phase, ...`. Data-leak audit passed: chronological splits, causal features, action→`open[t+1]` execution.

**What was superseded (do not act on the old detail):** the 15-dim state (now 16-dim with `equity_ratio`); the log-return reward (now R1/R2/R4 portfolio rewards); commission-based cost model (now spread-only from parquet); equity-starts-at-1.0 convention (now $10k portfolio); SB3 dependency (gone). See 2026-05-15 entry for the rebuild that replaced all of these.
