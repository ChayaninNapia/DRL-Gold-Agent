"""From-scratch Double DQN agent (no Stable-Baselines3).

Mirrors SB3 DQN defaults as closely as possible so that, given the same config,
training distributions are statistically comparable to the SB3 reference path in
`src/train.py`. Key choices that intentionally match SB3:

  - Optimizer: Adam with `eps=1.5e-4` (SB3 DQN default, not torch default 1e-8).
  - Loss: SmoothL1Loss (Huber) — SB3 DQN default.
  - Target update: hard copy every `target_update_interval` env steps
    (SB3 uses polyak=1.0 = hard copy).
  - Double DQN target: action selected by ONLINE net at s', evaluated by TARGET net.
  - ε-greedy: linear schedule over `exploration_fraction * total_timesteps`,
    then flat at `exploration_final_eps`.
  - Action selection: ε-greedy with random uniform; greedy via argmax of online Q.
  - Gradient clipping: global L2 norm cap at `max_grad_norm`.

Per-seed bit-equivalence with SB3 is NOT expected (different RNG ordering); the
goal is a statistical match across seeds. See journal entry for validation
protocol.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------- network ----------------------------


class QNetwork(nn.Module):
    """MLP Q-network: state -> Q-values for each discrete action."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU(inplace=True))
            last = h
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------- replay buffer ----------------------------


class ReplayBuffer:
    """Uniform-sample circular replay buffer backed by NumPy arrays.

    Stores transitions (obs, action, reward, next_obs, done). `done` here means
    the episode terminated naturally (env.terminated). It is used to mask the
    bootstrap term so the target = r when done.
    """

    def __init__(self, capacity: int, obs_dim: int, seed: int = 0):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.size = 0
        self.ptr = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = 1.0 if done else 0.0
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx], device=device),
            "actions": torch.as_tensor(self.actions[idx], device=device),
            "rewards": torch.as_tensor(self.rewards[idx], device=device),
            "next_obs": torch.as_tensor(self.next_obs[idx], device=device),
            "dones": torch.as_tensor(self.dones[idx], device=device),
        }

    def __len__(self) -> int:
        return self.size


# ---------------------------- exploration schedule ----------------------------


@dataclass
class EpsilonSchedule:
    """Linear interpolation from `start_eps` to `end_eps` over the first
    `fraction * total_steps`, then flat at `end_eps`."""

    start_eps: float
    end_eps: float
    fraction: float
    total_steps: int

    def __post_init__(self) -> None:
        self.decay_steps = max(1, int(self.fraction * self.total_steps))

    def value(self, step: int) -> float:
        if step >= self.decay_steps:
            return self.end_eps
        frac = step / self.decay_steps
        return self.start_eps + frac * (self.end_eps - self.start_eps)


# ---------------------------- DDQN agent ----------------------------


class DDQNAgent:
    """Double DQN with hard target updates and ε-greedy exploration."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: list[int],
        lr: float,
        gamma: float,
        buffer_size: int,
        batch_size: int,
        learning_starts: int,
        train_freq: int,
        gradient_steps: int,
        target_update_interval: int,
        exploration_fraction: float,
        exploration_initial_eps: float,
        exploration_final_eps: float,
        max_grad_norm: float,
        total_timesteps: int,
        device: torch.device,
        seed: int = 0,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts)
        self.train_freq = int(train_freq)
        self.gradient_steps = int(gradient_steps)
        self.target_update_interval = int(target_update_interval)
        self.max_grad_norm = float(max_grad_norm)
        self.device = device

        torch.manual_seed(seed)
        self.online = QNetwork(obs_dim, n_actions, hidden_sizes).to(device)
        self.target = QNetwork(obs_dim, n_actions, hidden_sizes).to(device)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)

        # SB3 DQN default: Adam with eps=1.5e-4 (NOT torch default 1e-8).
        self.optim = torch.optim.Adam(self.online.parameters(), lr=float(lr), eps=1.5e-4)
        self.loss_fn = nn.SmoothL1Loss()  # SB3 DQN default

        self.eps_schedule = EpsilonSchedule(
            start_eps=float(exploration_initial_eps),
            end_eps=float(exploration_final_eps),
            fraction=float(exploration_fraction),
            total_steps=int(total_timesteps),
        )
        self.replay = ReplayBuffer(buffer_size, obs_dim, seed=seed)
        self.act_rng = random.Random(seed + 1)
        self.np_rng = np.random.default_rng(seed + 2)

        # Diagnostics tracked per gradient step
        self.last_loss: float = float("nan")
        self.last_mean_q: float = float("nan")

    # ---- action selection ----

    def current_eps(self, step: int) -> float:
        return self.eps_schedule.value(step)

    def select_action(self, obs: np.ndarray, step: int, deterministic: bool = False) -> int:
        if not deterministic and self.act_rng.random() < self.current_eps(step):
            return self.act_rng.randrange(self.n_actions)
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(x)
            return int(q.argmax(dim=1).item())

    # ---- learning ----

    def _td_step(self) -> None:
        batch = self.replay.sample(self.batch_size, self.device)
        obs = batch["obs"]
        actions = batch["actions"].unsqueeze(1)
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        # Online Q at current state, gather by taken action.
        q_pred = self.online(obs).gather(1, actions).squeeze(1)

        # Double DQN target: a* = argmax_a Q_online(s', a); evaluate via target.
        with torch.no_grad():
            next_actions = self.online(next_obs).argmax(dim=1, keepdim=True)
            next_q = self.target(next_obs).gather(1, next_actions).squeeze(1)
            target = rewards + self.gamma * (1.0 - dones) * next_q

        loss = self.loss_fn(q_pred, target)
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.max_grad_norm)
        self.optim.step()

        self.last_loss = float(loss.detach().item())
        self.last_mean_q = float(q_pred.detach().mean().item())

    def maybe_learn(self, step: int) -> bool:
        """Perform `gradient_steps` Adam updates if conditions are met. Returns True
        if any update happened. Mirrors SB3 cadence: every `train_freq` env steps
        after `learning_starts`, run `gradient_steps` minibatch updates."""
        if step < self.learning_starts:
            return False
        if len(self.replay) < self.batch_size:
            return False
        if step % self.train_freq != 0:
            return False
        for _ in range(self.gradient_steps):
            self._td_step()
        return True

    def maybe_update_target(self, step: int) -> bool:
        if step > 0 and step % self.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
            return True
        return False

    # ---- checkpointing ----

    def save(self, path) -> None:
        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optim": self.optim.state_dict(),
        }, str(path))

    def load(self, path, strict: bool = True) -> None:
        ck = torch.load(str(path), map_location=self.device, weights_only=False)
        self.online.load_state_dict(ck["online"], strict=strict)
        self.target.load_state_dict(ck["target"], strict=strict)
        if "optim" in ck:
            self.optim.load_state_dict(ck["optim"])
