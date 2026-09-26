"""Torch trainer with the ch. 8 knobs: optimizer, clip, Polyak, BN lives in the model."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models import MLP, n_params, weight_grad_means


@dataclass
class TrainConfig:
    epochs: int = 8
    lr: float = 3e-3
    batch_size: int = 64
    optimizer: str = "sgd"
    momentum: float = 0.9
    weight_decay: float = 0.0
    grad_clip: float | None = None
    polyak: bool = False
    seed: int = 0


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    grad_l2: list[float] = field(default_factory=list)


@dataclass
class Metrics:
    train_acc: float
    val_acc: float
    test_acc: float
    gap: float
    n_params: int


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loader(X: np.ndarray, y: np.ndarray, batch: int, shuffle: bool) -> DataLoader:
    xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(xt, yt), batch_size=batch, shuffle=shuffle)


def make_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    p = model.parameters()
    name = cfg.optimizer.lower()
    if name == "sgd":
        return torch.optim.SGD(p, lr=cfg.lr, momentum=0.0, weight_decay=cfg.weight_decay)
    if name == "momentum":
        return torch.optim.SGD(
            p, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )
    if name == "nesterov":
        return torch.optim.SGD(
            p,
            lr=cfg.lr,
            momentum=cfg.momentum,
            nesterov=True,
            weight_decay=cfg.weight_decay,
        )
    if name == "adagrad":
        return torch.optim.Adagrad(p, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(p, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "adam":
        return torch.optim.Adam(p, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(name)


def _polyak_step(
    avg: torch.nn.Module, model: torch.nn.Module, n_seen: int
) -> int:
    # θ̄ ← n/(n+1) θ̄ + 1/(n+1) θ    (Polyak / running mean, §8.7.3)
    t = n_seen + 1
    with torch.no_grad():
        for a, p in zip(avg.parameters(), model.parameters()):
            a.mul_((t - 1) / t).add_(p, alpha=1.0 / t)
    return t


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for xb, yb in loader:
        logits = model(xb)
        total_loss += float(F.cross_entropy(logits, yb, reduction="sum"))
        correct += int((logits.argmax(1) == yb).sum())
        n += len(yb)
    if n == 0:
        return 0.0, 0.0
    return total_loss / n, correct / n


def train_classifier(
    model: torch.nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
) -> History:
    set_seed(cfg.seed)
    train_loader = _loader(X_train, y_train, cfg.batch_size, shuffle=True)
    val_loader = _loader(X_val, y_val, max(64, cfg.batch_size), shuffle=False)
    opt = make_optimizer(model, cfg)
    hist = History()
    avg = copy.deepcopy(model) if cfg.polyak else None
    n_seen = 0

    for _ in range(cfg.epochs):
        model.train()
        last_gn = 0.0
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            if cfg.grad_clip is not None:
                last_gn = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                )
            else:
                last_gn = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        last_gn += float(p.grad.detach().pow(2).sum())
                last_gn = last_gn**0.5
            opt.step()
            if avg is not None:
                n_seen = _polyak_step(avg, model, n_seen)
        tr_loss, tr_acc = evaluate(model, train_loader)
        va_loss, va_acc = evaluate(model, val_loader)
        hist.train_loss.append(tr_loss)
        hist.train_acc.append(tr_acc)
        hist.val_loss.append(va_loss)
        hist.val_acc.append(va_acc)
        hist.grad_l2.append(last_gn)

    if avg is not None:
        model.load_state_dict(avg.state_dict())
    return hist


def metrics_for(
    model: torch.nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Metrics:
    tr = evaluate(model, _loader(X_train, y_train, 256, False))
    va = evaluate(model, _loader(X_val, y_val, 256, False))
    te = evaluate(model, _loader(X_test, y_test, 256, False))
    return Metrics(tr[1], va[1], te[1], tr[1] - te[1], n_params(model))


def make_mlp(
    in_dim: int,
    n_classes: int,
    hidden: tuple[int, ...] = (128, 128),
    activation: str = "relu",
    batch_norm: bool = False,
    init: str = "xavier",
    seed: int = 0,
) -> MLP:
    set_seed(seed)
    return MLP(in_dim, n_classes, hidden, activation, batch_norm, init)


def one_batch_grad_profile(
    model: torch.nn.Module, X: np.ndarray, y: np.ndarray, n: int = 64
) -> list[tuple[str, float]]:
    """One backward pass, mean |grad| per weight matrix."""
    model.train()
    xb = torch.as_tensor(X[:n], dtype=torch.float32)
    yb = torch.as_tensor(y[:n], dtype=torch.long)
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(xb), yb).backward()
    return weight_grad_means(model)
