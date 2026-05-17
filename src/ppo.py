"""Proximal Policy Optimization (PPO, Schulman 2017) — clipped surrogate objective.

Reuses `ActorCritic`, `Rollout`, and `compute_gae` from `src/a2c.py` (same MDP, same
shared-trunk architecture, same advantage estimator). Only the update rule differs:

  - K SGD epochs over each rollout (default 10) — sample efficiency vs A2C's 1 epoch.
  - Minibatch sampling within each epoch — random permutation, fixed minibatch size.
  - Clipped surrogate policy loss:
        L^CLIP = E[ min( r_t · A_t,  clip(r_t, 1-ε, 1+ε) · A_t ) ]
      where r_t = π_new(a_t|s_t) / π_old(a_t|s_t).
  - Optional value clipping (`clip_range_vf`): bound V_new to V_old ± ε_v.
  - Optional KL early-stop within an update (`target_kl`): break out of remaining
    epochs if mean approx-KL exceeds target.
  - Same combined loss form as A2C: L = L^CLIP_pol + value_coef · L_V − entropy_coef · H.

`PPOAgent.select_action(obs, step=0, deterministic=...)` matches DDQN/A2C signature
so `evaluate_policy_per_session()` works without changes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from a2c import ActorCritic, Rollout, _inv_freq_weights, compute_gae


class PPOAgent:
    """PPO with shared-trunk actor-critic. One rollout = one full episode in our setup."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: list[int],
        lr: float,
        gamma: float,
        gae_lambda: float,
        clip_range: float,
        clip_range_vf: float | None,
        value_coef: float,
        entropy_coef: float,
        max_grad_norm: float,
        n_epochs: int,
        minibatch_size: int,
        target_kl: float | None,
        normalize_advantage: bool,
        device: torch.device,
        seed: int = 0,
        bc_coef: float = 0.0,
        bc_anneal_steps: int = 0,
        bc_class_weight: bool = False,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_range = float(clip_range)
        self.clip_range_vf = None if clip_range_vf is None else float(clip_range_vf)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.n_epochs = int(n_epochs)
        self.minibatch_size = int(minibatch_size)
        self.target_kl = None if target_kl is None else float(target_kl)
        self.normalize_advantage = bool(normalize_advantage)
        self.device = device

        # BC warm-start (Phase 1d). bc_coef > 0 activates an auxiliary CE loss on
        # expert action labels supplied per-bar in the rollout. Coefficient anneals
        # linearly bc_coef -> 0 over bc_anneal_steps env steps.
        self.bc_coef_initial = float(bc_coef)
        self.bc_anneal_steps = int(bc_anneal_steps)
        # B1 (2026-05-16): inverse-frequency class weighting on the BC CE loss.
        # Computed once per rollout from the expert label frequencies (a property
        # of the expert, not the policy — so it does NOT go stale across PPO's K
        # epochs, unlike advantage stats). Same rule as A2C via _inv_freq_weights.
        self.bc_class_weight = bool(bc_class_weight)

        torch.manual_seed(seed)
        self.net = ActorCritic(obs_dim, n_actions, hidden_sizes).to(device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=float(lr))
        # numpy RNG for minibatch shuffling; separate from torch global state
        self.rng = np.random.default_rng(seed)

        # Diagnostics (last update)
        self.last_policy_loss: float = float("nan")
        self.last_value_loss: float = float("nan")
        self.last_entropy: float = float("nan")
        self.last_total_loss: float = float("nan")
        self.last_explained_var: float = float("nan")
        self.last_approx_kl: float = float("nan")
        self.last_clip_fraction: float = float("nan")
        self.last_n_epochs_run: int = 0
        self.last_bc_loss: float = float("nan")
        self.last_bc_coef: float = float("nan")

    def _bc_coef(self, global_step: int) -> float:
        if self.bc_coef_initial <= 0.0 or self.bc_anneal_steps <= 0:
            return 0.0
        frac = max(0.0, 1.0 - float(global_step) / float(self.bc_anneal_steps))
        return self.bc_coef_initial * frac

    # ---- action selection (drop-in compatible with DDQN/A2C) ----

    def select_action(self, obs: np.ndarray, step: int = 0, deterministic: bool = False) -> int:
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, _ = self.net(x)
            if deterministic:
                return int(logits.argmax(dim=1).item())
            return int(Categorical(logits=logits).sample().item())

    def act_and_evaluate(self, obs: np.ndarray) -> tuple[int, float, float]:
        """Sample action and return (action, log_prob, value) in one forward pass."""
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, value = self.net(x)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return int(action.item()), float(log_prob.item()), float(value.item())

    # ---- learning ----

    def update(self, rollout: Rollout, last_value: float = 0.0,
               global_step: int = 0) -> None:
        n = len(rollout)
        if n == 0:
            return

        rewards = np.asarray(rollout.rewards, dtype=np.float64)
        values_old = np.asarray(rollout.values, dtype=np.float64)
        dones = np.asarray(rollout.dones, dtype=np.float64)
        log_probs_old = np.asarray(rollout.log_probs, dtype=np.float64)

        advantages, returns = compute_gae(
            rewards, values_old, dones, last_value=last_value,
            gamma=self.gamma, gae_lambda=self.gae_lambda,
        )

        obs_t = torch.as_tensor(np.stack(rollout.obs), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(rollout.actions, dtype=torch.int64, device=self.device)
        log_probs_old_t = torch.as_tensor(log_probs_old, dtype=torch.float32, device=self.device)
        values_old_t = torch.as_tensor(values_old, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        # BC: assemble expert tensor and active mask once per rollout
        bc_coef_now = self._bc_coef(global_step)
        bc_active = bc_coef_now > 0.0 and len(rollout.expert_actions) == n
        ce_weight = None
        if bc_active:
            expert_t = torch.as_tensor(rollout.expert_actions, dtype=torch.int64, device=self.device)
            if self.bc_class_weight:
                # Inverse-freq weights from the whole rollout's valid expert
                # labels (constant across the K epochs — not policy-dependent).
                ce_weight = _inv_freq_weights(
                    expert_t[expert_t >= 0], self.n_actions, self.device
                )
        else:
            expert_t = None

        # Diagnostics accumulators across all minibatches x epochs
        pol_losses: list[float] = []
        val_losses: list[float] = []
        entropies: list[float] = []
        kls: list[float] = []
        clip_fracs: list[float] = []
        bc_losses: list[float] = []
        epochs_run = 0
        stop_early = False

        for epoch in range(self.n_epochs):
            # Shuffle indices for minibatch SGD
            idx = self.rng.permutation(n)
            for start in range(0, n, self.minibatch_size):
                mb = idx[start : start + self.minibatch_size]
                mb_t = torch.as_tensor(mb, dtype=torch.long, device=self.device)
                obs_mb = obs_t[mb_t]
                actions_mb = actions_t[mb_t]
                log_probs_old_mb = log_probs_old_t[mb_t]
                values_old_mb = values_old_t[mb_t]
                adv_mb = advantages_t[mb_t]
                returns_mb = returns_t[mb_t]

                if self.normalize_advantage and adv_mb.numel() > 1:
                    adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)

                logits, values_new = self.net(obs_mb)
                dist = Categorical(logits=logits)
                log_probs_new = dist.log_prob(actions_mb)
                entropy = dist.entropy().mean()

                # Clipped surrogate policy loss
                ratio = torch.exp(log_probs_new - log_probs_old_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (optionally clipped)
                if self.clip_range_vf is None:
                    value_loss = F.mse_loss(values_new, returns_mb)
                else:
                    values_clipped = values_old_mb + torch.clamp(
                        values_new - values_old_mb, -self.clip_range_vf, self.clip_range_vf,
                    )
                    vl_unclipped = (values_new - returns_mb).pow(2)
                    vl_clipped = (values_clipped - returns_mb).pow(2)
                    value_loss = 0.5 * torch.max(vl_unclipped, vl_clipped).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # BC auxiliary loss on the minibatch (if active and labels present)
                bc_loss_val = float("nan")
                if bc_active:
                    expert_mb = expert_t[mb_t]
                    bc_mask = expert_mb >= 0
                    if int(bc_mask.sum().item()) > 0:
                        bc_logits_mb = logits[bc_mask]
                        bc_targets_mb = expert_mb[bc_mask]
                        bc_loss = F.cross_entropy(bc_logits_mb, bc_targets_mb, weight=ce_weight)
                        loss = loss + bc_coef_now * bc_loss
                        bc_loss_val = float(bc_loss.detach().item())

                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optim.step()

                with torch.no_grad():
                    # Approx KL (Schulman 2020 blog: k3 estimator is unbiased+positive).
                    # We use the simpler k1 = mean(log_old - log_new) for matching SB3/baselines.
                    approx_kl = (log_probs_old_mb - log_probs_new).mean()
                    clip_frac = ((ratio - 1.0).abs() > self.clip_range).float().mean()

                pol_losses.append(float(policy_loss.detach().item()))
                val_losses.append(float(value_loss.detach().item()))
                entropies.append(float(entropy.detach().item()))
                kls.append(float(approx_kl.item()))
                clip_fracs.append(float(clip_frac.item()))
                if not (bc_loss_val != bc_loss_val):  # not NaN
                    bc_losses.append(bc_loss_val)

            epochs_run = epoch + 1
            if self.target_kl is not None and len(kls) > 0:
                # Average KL over this epoch's minibatches
                n_mb_per_epoch = max(1, (n + self.minibatch_size - 1) // self.minibatch_size)
                last_epoch_kl = float(np.mean(kls[-n_mb_per_epoch:]))
                if last_epoch_kl > 1.5 * self.target_kl:
                    stop_early = True
                    break

        # Diagnostics
        self.last_policy_loss = float(np.mean(pol_losses)) if pol_losses else float("nan")
        self.last_value_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        self.last_entropy = float(np.mean(entropies)) if entropies else float("nan")
        self.last_total_loss = self.last_policy_loss + self.value_coef * self.last_value_loss - self.entropy_coef * self.last_entropy
        self.last_approx_kl = float(np.mean(kls)) if kls else float("nan")
        self.last_clip_fraction = float(np.mean(clip_fracs)) if clip_fracs else float("nan")
        self.last_n_epochs_run = epochs_run
        self.last_bc_loss = float(np.mean(bc_losses)) if bc_losses else float("nan")
        self.last_bc_coef = bc_coef_now

        # Explained variance on the full rollout (one final forward pass for clarity)
        with torch.no_grad():
            _, values_eval = self.net(obs_t)
            var_returns = returns_t.var()
            if var_returns > 1e-12:
                ev = 1.0 - (returns_t - values_eval).var() / var_returns
                self.last_explained_var = float(ev.item())
            else:
                self.last_explained_var = float("nan")

    # ---- checkpointing ----

    def save(self, path) -> None:
        torch.save({
            "net": self.net.state_dict(),
            "optim": self.optim.state_dict(),
        }, str(path))

    def load(self, path, strict: bool = True) -> None:
        ck = torch.load(str(path), map_location=self.device, weights_only=False)
        self.net.load_state_dict(ck["net"], strict=strict)
        if "optim" in ck:
            self.optim.load_state_dict(ck["optim"])
