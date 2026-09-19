"""Models used in the ch. 7 experiments.

Biases are left unregularized (ch. 7.1): a bias is one number per unit and
regularizing it underfits more than it cuts variance.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def decay_param_groups(
    model: nn.Module, weight_decay: float
) -> list[dict]:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith("bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def weight_l1(model: nn.Module) -> torch.Tensor:
    terms = [
        p.abs().sum()
        for n, p in model.named_parameters()
        if p.ndim > 1 and p.requires_grad
    ]
    return torch.stack(terms).sum() if terms else torch.tensor(0.0)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        hidden: tuple[int, ...] = (128, 128),
        dropout: float = 0.0,
        batch_norm: bool = False,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        act: nn.Module = nn.Tanh() if activation == "tanh" else nn.ReLU()
        layers: list[nn.Module] = []
        prev = in_dim
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            if batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(act.__class__())
            if dropout > 0:
                # inverted dropout: scales at train time, identity at eval
                layers.append(nn.Dropout(dropout))
            prev = width
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_classes)

    def forward(
        self, x: torch.Tensor, return_hidden: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = x.reshape(x.size(0), -1)
        h = self.backbone(x)
        logits = self.head(h)
        if return_hidden:
            return logits, h
        return logits


class TinyCNN(nn.Module):
    """§7.9.1 parameter sharing: same 3x3 kernel at every spatial location."""

    def __init__(self, n_classes: int = 10, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(16 * 4 * 4, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.reshape(-1, 1, 8, 8)
        h = self.conv(x)
        h = self.drop(h.reshape(h.size(0), -1))
        return self.head(h)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def fgsm(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """§7.13: one-step adversarial example, sign of ∇_x J."""
    x = x.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    assert x.grad is not None
    adv = x + eps * x.grad.sign()
    return adv.detach().clamp(0.0, 1.0)
