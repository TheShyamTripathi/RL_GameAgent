"""
replay_buffer.py
----------------
Numpy-only experience replay (no torch import, so it is easy to unit-test):

* NStepAccumulator         - turns 1-step transitions into n-step transitions
* ReplayBuffer             - uniform circular buffer
* SumTree                  - O(log n) proportional sampling structure
* PrioritizedReplayBuffer  - Prioritized Experience Replay (Schaul et al., 2016)
"""
from __future__ import annotations

from collections import deque

import numpy as np


class NStepAccumulator:
    """
    Builds n-step transitions  (s_t, a_t, R, s_{t+k}, terminated, gamma^k)  where
    R = sum_{i<k} gamma^i r_{t+i}  and k <= n (shorter at the end of an episode).
    The stored discount gamma^k is used to bootstrap: target = R + gamma^k * V(s_{t+k}).
    """

    def __init__(self, n_step: int = 1, gamma: float = 0.99):
        self.n = max(1, int(n_step))
        self.gamma = gamma
        self.window: deque = deque()

    def push(self, state, action, reward, next_state, terminated: bool, episode_over: bool):
        self.window.append((state, action, reward, next_state, terminated))
        out = []
        if len(self.window) == self.n:
            out.append(self._build())
            self.window.popleft()
        if episode_over:                      # flush what is left of the window
            while self.window:
                out.append(self._build())
                self.window.popleft()
        return out

    def reset(self) -> None:
        self.window.clear()

    def _build(self):
        s0, a0 = self.window[0][0], self.window[0][1]
        ret = 0.0
        for i, (_, _, r, _, _) in enumerate(self.window):
            ret += (self.gamma ** i) * r
        _, _, _, s_n, terminated = self.window[-1]
        return s0, a0, ret, s_n, float(terminated), self.gamma ** len(self.window)


class ReplayBuffer:
    """Uniform circular replay buffer with built-in n-step support."""

    def __init__(self, capacity: int, state_dim: int, n_step: int = 1, gamma: float = 0.99):
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.discounts = np.ones((self.capacity, 1), dtype=np.float32)
        self.nstep = NStepAccumulator(n_step, gamma)

    # ------------------------------------------------------------------ write
    def add(self, state, action, reward, next_state, terminated: bool, episode_over: bool) -> None:
        for tr in self.nstep.push(state, action, reward, next_state, terminated, episode_over):
            self._store(*tr)

    def _store(self, s, a, ret, s_n, terminated, discount) -> None:
        i = self.ptr
        self.states[i] = s
        self.actions[i] = a
        self.rewards[i] = ret
        self.next_states[i] = s_n
        self.dones[i] = terminated
        self.discounts[i] = discount
        self._on_store(i)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _on_store(self, idx: int) -> None:       # hook for subclasses
        pass

    # ------------------------------------------------------------------- read
    def sample(self, batch_size: int, beta: float | None = None) -> dict:
        idx = np.random.randint(0, self.size, size=batch_size)
        return self._gather(idx, np.ones(batch_size, dtype=np.float32))

    def _gather(self, idx: np.ndarray, weights: np.ndarray) -> dict:
        return {
            "states": self.states[idx], "actions": self.actions[idx],
            "rewards": self.rewards[idx], "next_states": self.next_states[idx],
            "dones": self.dones[idx], "discounts": self.discounts[idx],
            "weights": weights.reshape(-1, 1).astype(np.float32), "indices": idx,
        }

    def update_priorities(self, indices, td_errors) -> None:   # no-op for uniform replay
        pass

    def __len__(self) -> int:
        return self.size


class SumTree:
    """Binary sum tree; leaves hold priorities. Capacity is rounded up to a power of two."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.leaves = 1 << max(0, (self.capacity - 1).bit_length())
        self.tree = np.zeros(2 * self.leaves, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def get(self, idx) -> np.ndarray:
        return self.tree[self.leaves + np.asarray(idx, dtype=np.int64)]

    def set(self, idx, priorities) -> None:
        pos = np.asarray(idx, dtype=np.int64) + self.leaves
        self.tree[pos] = priorities
        pos = np.unique(pos >> 1)
        while pos.size and pos[0] >= 1:
            self.tree[pos] = self.tree[2 * pos] + self.tree[2 * pos + 1]
            pos = np.unique(pos >> 1)

    def find(self, values) -> np.ndarray:
        """Vectorised prefix-sum search: returns leaf (data) indices."""
        vals = np.asarray(values, dtype=np.float64).copy()
        idx = np.ones(len(vals), dtype=np.int64)
        while idx[0] < self.leaves:
            left = 2 * idx
            left_sum = self.tree[left]
            go_right = vals > left_sum
            vals = np.where(go_right, vals - left_sum, vals)
            idx = np.where(go_right, left + 1, left)
        return idx - self.leaves


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Proportional PER:  P(i) = p_i^alpha / sum_k p_k^alpha,   w_i = (N * P(i))^-beta / max_j w_j
    New transitions get the current max priority so each is replayed at least once.
    """

    def __init__(self, capacity, state_dim, n_step=1, gamma=0.99, alpha=0.6, eps=1e-6):
        super().__init__(capacity, state_dim, n_step, gamma)
        self.alpha = alpha
        self.eps = eps
        self.tree = SumTree(self.capacity)
        self.max_priority = 1.0

    def _on_store(self, idx: int) -> None:
        self.tree.set([idx], [self.max_priority])

    def sample(self, batch_size: int, beta: float | None = 0.4) -> dict:
        beta = 0.4 if beta is None else beta
        total = self.tree.total
        segment = total / batch_size                       # stratified sampling
        values = (np.arange(batch_size) + np.random.random(batch_size)) * segment
        idx = np.minimum(self.tree.find(values), self.size - 1)
        probs = np.maximum(self.tree.get(idx) / total, 1e-12)
        weights = (self.size * probs) ** (-beta)
        weights = weights / weights.max()
        return self._gather(idx, weights.astype(np.float32))

    def update_priorities(self, indices, td_errors) -> None:
        p = (np.abs(np.asarray(td_errors, dtype=np.float64)) + self.eps) ** self.alpha
        self.tree.set(indices, p)
        self.max_priority = max(self.max_priority, float(p.max()))
