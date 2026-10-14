"""
networks.py
-----------
Q-network: a plain MLP head or a Dueling head (Wang et al., 2016)
    Q(s,a) = V(s) + A(s,a) - mean_a' A(s,a')
"""
from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden=(128, 128), dueling: bool = True):
        super().__init__()
        self.dueling = dueling

        layers, in_dim = [], state_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.feature = nn.Sequential(*layers)

        if dueling:
            self.value = nn.Linear(in_dim, 1)
            self.advantage = nn.Linear(in_dim, n_actions)
        else:
            self.head = nn.Linear(in_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature(x)
        if self.dueling:
            adv = self.advantage(x)
            return self.value(x) + adv - adv.mean(dim=1, keepdim=True)
        return self.head(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
