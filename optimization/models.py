"""Small nets for the ch. 8 notebook.

Init schemes are first-class: zero / tiny / xavier / he / huge.
Batch-norm is optional (ch. 8.7.1).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def init_linear(module: nn.Linear, scheme: str) -> None:
    if scheme == "zero":
        nn.init.zeros_(module.weight)
    elif scheme == "tiny":
        nn.init.normal_(module.weight, mean=0.0, std=1e-2)
    elif scheme == "xavier":
        nn.init.xavier_uniform_(module.weight)
    elif scheme == "he":
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
    elif scheme == "huge":
        nn.init.normal_(module.weight, mean=0.0, std=1.0)
    else:
        raise ValueError(scheme)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def apply_init(model: nn.Module, scheme: str) -> nn.Module:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            init_linear(m, scheme)
    return model


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        hidden: tuple[int, ...] = (128, 128),
        activation: str = "relu",
        batch_norm: bool = False,
        init: str = "xavier",
    ) -> None:
        super().__init__()
        act: type[nn.Module]
        if activation == "tanh":
            act = nn.Tanh
        elif activation == "sigmoid":
            act = nn.Sigmoid
        else:
            act = nn.ReLU
        layers: list[nn.Module] = []
        prev = in_dim
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            if batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(act())
            prev = width
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_classes)
        apply_init(self, init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x.reshape(x.size(0), -1)))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def weight_grad_means(model: nn.Module) -> list[tuple[str, float]]:
    """Mean |grad| per weight matrix, input-layer first. For vanishing-grad bars."""
    out: list[tuple[str, float]] = []
    for name, p in model.named_parameters():
        if p.ndim > 1 and p.grad is not None:
            out.append((name, float(p.grad.detach().abs().mean())))
    return out
