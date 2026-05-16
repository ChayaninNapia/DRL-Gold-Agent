"""Advantage Actor-Critic (A2C) — on-policy, shared trunk, single-step update per rollout.

Design choices (locked in with user before implementation):
  - Shared MLP trunk + 2 heads (policy logits over 3 discrete actions, scalar value).
  - 1 rollout = 1 full trading day (episode-aligned). Update once per completed episode.
  - GAE(λ) for advantage estimation. λ = `gae_lambda`.
  - Advantage normalization per rollout (configurable, default ON) — handles our
    small reward magnitudes (~±0.001 per bar) without changing the env/reward.
  - Combined loss: policy gradient − value_coef × value MSE + entropy_coef × entropy.

`A2CAgent.select_action(obs, step=..., deterministic=...)` keeps the same signature
as `DDQNAgent.select_action()` so that `evaluate_policy_per_session()` in train.py
works for both agents without changes (the `step` arg is ignored here).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ---------------------------- network ----------------------------


class ActorCritic(nn.Module):
    """Shared MLP trunk + policy head (logits) + value head (scalar)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.Tanh())  # Tanh is standard for on-policy; smoother gradients than ReLU
            last = h
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last, n_actions)
        self.value_head = nn.Linear(last, 1)

        # Smaller init on policy head so early policy is near-uniform (encourages exploration
        # without relying on entropy bonus alone). Value head can stay default.
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


# ---------------------------- rollout buffer ----------------------------


@dataclass
class Rollout:
    """One episode worth of on-policy transitions, stored as Python lists.

    A2C needs (obs, action, log_prob, value, reward, done) for the update.
    We also store `next_position` from env info — needed for trade reconstruction
    (positions != actions on the final bar where the env force-flattens).
    """
    obs: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    next_positions: list[float] = field(default_factory=list)
    expert_actions: list[int] = field(default_factory=list)  # for BC loss; -1 if absent

    def __len__(self) -> int:
        return len(self.obs)

    def add(self, obs: np.ndarray, action: int, log_prob: float, value: float,
            reward: float, done: bool, next_position: float,
            expert_action: int = -1) -> None:
        self.obs.append(obs.copy())
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.next_positions.append(float(next_position))
        self.expert_actions.append(int(expert_action))


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation (Schulman 2016).

    δ_t      = r_t + γ V(s_{t+1}) (1 − done_t) − V(s_t)
    A_t      = δ_t + γ λ (1 − done_t) A_{t+1}
    return_t = A_t + V(s_t)

    For 1-episode rollout where the episode terminated, last_value should be 0
    (no bootstrap past terminal). dones[t] = True only at the final step.
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float64)
    last_adv = 0.0
    for t in reversed(range(n)):
        next_value = last_value if t == n - 1 else values[t + 1]
        next_nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_adv = delta + gamma * gae_lambda * next_nonterminal * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns


# ---------------------------- agent ----------------------------


class A2CAgent:
    """Advantage Actor-Critic. One Adam step per completed episode."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: list[int],
        lr: float,
        gamma: float,
        gae_lambda: float,
        value_coef: float,
        entropy_coef: float,
        max_grad_norm: float,
        normalize_advantage: bool,
        device: torch.device,
        seed: int = 0,
        bc_coef: float = 0.0,
        bc_anneal_steps: int = 0,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.normalize_advantage = bool(normalize_advantage)
        self.device = device

        # BC warm-start (Phase 1d). bc_coef > 0 activates an auxiliary cross-entropy
        # loss on expert action labels supplied in the rollout. Coefficient anneals
        # linearly from bc_coef -> 0 over bc_anneal_steps env steps.
        self.bc_coef_initial = float(bc_coef)
        self.bc_anneal_steps = int(bc_anneal_steps)

        torch.manual_seed(seed)
        self.net = ActorCritic(obs_dim, n_actions, hidden_sizes).to(device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=float(lr))

        # Diagnostics tracked across updates
        self.last_policy_loss: float = float("nan")
        self.last_value_loss: float = float("nan")
        self.last_entropy: float = float("nan")
        self.last_total_loss: float = float("nan")
        self.last_explained_var: float = float("nan")
        self.last_bc_loss: float = float("nan")
        self.last_bc_coef: float = float("nan")

    # ---- action selection ----

    def select_action(self, obs: np.ndarray, step: int = 0, deterministic: bool = False) -> int:
        """Drop-in compatible signature with DDQNAgent.select_action. `step` is ignored
        (no ε-schedule). Stochastic = sample from Categorical; deterministic = argmax."""
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, _ = self.net(x)
            if deterministic:
                return int(logits.argmax(dim=1).item())
            dist = Categorical(logits=logits)
            return int(dist.sample().item())

    def evaluate_action(self, obs: np.ndarray, action: int) -> tuple[float, float]:
        """Return (log_prob, value) for a given (obs, action). Used during rollout
        collection to log probs at the time of action."""
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, value = self.net(x)
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(torch.as_tensor([action], device=self.device))
            return float(log_prob.item()), float(value.item())

    def act_and_evaluate(self, obs: np.ndarray) -> tuple[int, float, float]:
        """Sample action and return (action, log_prob, value) in a single forward pass.
        More efficient than select_action + evaluate_action during training rollout."""
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, value = self.net(x)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return int(action.item()), float(log_prob.item()), float(value.item())

    def value_only(self, obs: np.ndarray) -> float:
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, value = self.net(x)
            return float(value.item())

    # ---- learning ----

    def _bc_coef(self, global_step: int) -> float:
        """Linear-anneal BC coefficient from bc_coef_initial -> 0 over bc_anneal_steps.
        Returns 0.0 if bc_coef_initial == 0 or after anneal completes."""
        if self.bc_coef_initial <= 0.0 or self.bc_anneal_steps <= 0:
            return 0.0
        frac = max(0.0, 1.0 - float(global_step) / float(self.bc_anneal_steps))
        return self.bc_coef_initial * frac

    def update(self, rollout: Rollout, last_value: float = 0.0,
               global_step: int = 0) -> None:
        """One gradient step on the combined A2C loss for the whole rollout.

        Combined loss: L = -E[A * logpi] + value_coef * MSE(V, R) - entropy_coef * H(pi)
                         + bc_coef(step) * CE(pi(s), a_expert)   [if expert actions present]
        """
        if len(rollout) == 0:
            return

        rewards = np.asarray(rollout.rewards, dtype=np.float64)
        values_old = np.asarray(rollout.values, dtype=np.float64)
        dones = np.asarray(rollout.dones, dtype=np.float64)

        advantages, returns = compute_gae(
            rewards, values_old, dones, last_value=last_value,
            gamma=self.gamma, gae_lambda=self.gae_lambda,
        )

        # Move to tensors on device
        obs_t = torch.as_tensor(np.stack(rollout.obs), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(rollout.actions, dtype=torch.int64, device=self.device)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        if self.normalize_advantage and adv_t.numel() > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # Forward (fresh) to get current logits + values for the whole rollout
        logits, values_new = self.net(obs_t)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()

        policy_loss = -(adv_t * log_probs).mean()
        value_loss = F.mse_loss(values_new, returns_t)
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        # BC auxiliary loss (only if expert actions present AND bc_coef > 0 at this step)
        bc_coef_now = self._bc_coef(global_step)
        bc_loss_val = float("nan")
        if bc_coef_now > 0.0 and len(rollout.expert_actions) == len(rollout.obs):
            expert_t = torch.as_tensor(rollout.expert_actions, dtype=torch.int64, device=self.device)
            # Filter out -1 (no-expert) entries
            mask = expert_t >= 0
            if int(mask.sum().item()) > 0:
                bc_logits = logits[mask]
                bc_targets = expert_t[mask]
                bc_loss = F.cross_entropy(bc_logits, bc_targets)
                loss = loss + bc_coef_now * bc_loss
                bc_loss_val = float(bc_loss.detach().item())

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optim.step()

        # Diagnostics
        self.last_policy_loss = float(policy_loss.detach().item())
        self.last_value_loss = float(value_loss.detach().item())
        self.last_entropy = float(entropy.detach().item())
        self.last_total_loss = float(loss.detach().item())
        self.last_bc_loss = bc_loss_val
        self.last_bc_coef = bc_coef_now
        # Explained variance: 1 − Var(R − V) / Var(R). Useful sanity check.
        with torch.no_grad():
            var_returns = returns_t.var()
            if var_returns > 1e-12:
                ev = 1.0 - (returns_t - values_new).var() / var_returns
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
