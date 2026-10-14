"""
agent.py
--------
DQN agent with switchable extensions (all driven by `TrainConfig`):

* Double DQN             - decoupled action selection / evaluation
* Dueling architecture   - see networks.py
* Prioritized replay     - importance-sampling weighted loss + priority updates
* N-step returns         - discount stored per transition (gamma^k)
* Soft target updates, gradient clipping, Huber loss, epsilon-greedy exploration
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import configs
from configs import TrainConfig
from networks import QNetwork
from replay_buffer import PrioritizedReplayBuffer, ReplayBuffer


# ----------------------------------------------------------------------------
# Checkpoint safety helpers
# ----------------------------------------------------------------------------
def _to_builtin(obj):
    """Recursively convert NumPy scalars/arrays to plain Python types.
    Keeps checkpoints loadable with torch.load(weights_only=True)."""
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _allow_numpy_globals() -> None:
    """Allowlist the NumPy classes that old checkpoints may still contain."""
    try:
        from numpy._core.multiarray import scalar as np_scalar   # NumPy 2.x
    except ImportError:
        from numpy.core.multiarray import scalar as np_scalar    # NumPy 1.x
    torch.serialization.add_safe_globals([np_scalar])


_allow_numpy_globals()


class DQNAgent:
    def __init__(self, state_dim: int, n_actions: int, cfg: TrainConfig, for_inference: bool = False):
        self.state_dim, self.n_actions, self.cfg = state_dim, n_actions, cfg
        self.device = configs.get_device()

        hidden = tuple(cfg.hidden)
        self.policy_net = QNetwork(state_dim, n_actions, hidden, cfg.dueling).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions, hidden, cfg.dueling).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.epsilon = 0.0 if for_inference else cfg.eps_start
        self.beta = cfg.per_beta_start
        self.learn_steps = 0
        self.buffer = None
        self.optimizer = None

        if not for_inference:
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
            common = dict(capacity=cfg.buffer_size, state_dim=state_dim, n_step=cfg.n_step, gamma=cfg.gamma)
            self.buffer = (PrioritizedReplayBuffer(alpha=cfg.per_alpha, **common)
                           if cfg.prioritized else ReplayBuffer(**common))

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def act(self, state, greedy: bool = False) -> int:
        if not greedy and np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        s = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)
        return int(self.policy_net(s).argmax(dim=1).item())

    def observe(self, state, action, reward, next_state, terminated: bool, episode_over: bool) -> None:
        self.buffer.add(state, action, reward, next_state, terminated, episode_over)

    def end_episode(self, progress: float) -> None:
        """Called once per episode: decays epsilon and anneals the PER beta towards 1."""
        self.epsilon = max(self.cfg.eps_end, self.epsilon * self.cfg.eps_decay)
        b0 = self.cfg.per_beta_start
        self.beta = b0 + (1.0 - b0) * min(1.0, progress)

    # ---------------------------------------------------------------- learning
    def learn(self) -> dict | None:
        """One gradient step. Returns diagnostics, or None while warming up."""
        cfg = self.cfg
        if self.buffer is None or len(self.buffer) < max(cfg.batch_size, cfg.warmup_steps):
            return None

        batch = self.buffer.sample(cfg.batch_size, beta=self.beta)
        t = lambda key: torch.as_tensor(batch[key], device=self.device)
        s, a, r, s2 = t("states"), t("actions"), t("rewards"), t("next_states")
        done, discount, weights = t("dones"), t("discounts"), t("weights")

        q = self.policy_net(s).gather(1, a)
        with torch.no_grad():
            if cfg.double_dqn:
                best = self.policy_net(s2).argmax(dim=1, keepdim=True)
                next_q = self.target_net(s2).gather(1, best)
            else:
                next_q = self.target_net(s2).max(dim=1, keepdim=True)[0]
            target = r + discount * next_q * (1.0 - done)

        td_error = target - q
        loss = (weights * F.smooth_l1_loss(q, target, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), cfg.grad_clip)
        self.optimizer.step()

        if cfg.prioritized:
            self.buffer.update_priorities(batch["indices"], td_error.detach().abs().squeeze(1).cpu().numpy())

        with torch.no_grad():                                    # Polyak averaging
            for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
                tp.mul_(1.0 - cfg.tau).add_(pp, alpha=cfg.tau)

        self.learn_steps += 1
        return {"loss": float(loss.item()), "mean_q": float(q.mean().item())}

    # ------------------------------------------------------------ persistence
    def save(self, path: str, extra: dict | None = None) -> None:
        checkpoint = _to_builtin({
            "state_dict": self.policy_net.state_dict(),   # tensors are left untouched
            "state_dim": int(self.state_dim),
            "n_actions": int(self.n_actions),
            "cfg": self.cfg.to_dict(),
            "extra": extra or {},
        })
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str) -> "DQNAgent":
        ckpt = torch.load(path, map_location=configs.get_device(), weights_only=True)
        cfg = TrainConfig.from_dict(ckpt["cfg"])
        agent = cls(ckpt["state_dim"], ckpt["n_actions"], cfg, for_inference=True)
        agent.policy_net.load_state_dict(ckpt["state_dict"])
        agent.target_net.load_state_dict(ckpt["state_dict"])
        agent.policy_net.eval()
        return agent