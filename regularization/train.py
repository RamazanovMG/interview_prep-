"""Shared training loop: penalties, noise, early stopping, FGSM, bagging."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models import MLP, decay_param_groups, fgsm, weight_l1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TrainConfig:
    epochs: int = 80
    lr: float = 0.05
    batch_size: int = 32
    weight_decay: float = 0.0
    l1: float = 0.0
    activation_l1: float = 0.0
    input_noise: float = 0.0
    label_smoothing: float = 0.0
    early_stop_patience: int | None = None
    early_stop_min_epoch: int = 10
    adversarial_eps: float = 0.0
    restore_best: bool = False
    seed: int = 0
    momentum: float = 0.9
    optimizer: str = "sgd"


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    best_epoch: int = 0
    stopped_epoch: int = 0


@dataclass
class Metrics:
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    test_loss: float
    test_acc: float
    gap: float


def _loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    xt = torch.from_numpy(np.ascontiguousarray(X))
    if np.issubdtype(y.dtype, np.floating):
        yt = torch.from_numpy(np.ascontiguousarray(y))
    else:
        yt = torch.from_numpy(np.ascontiguousarray(y)).long()
    return DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    classification = True
    for xb, yb in loader:
        logits = model(xb)
        classification = logits.ndim > 1 and logits.shape[-1] > 1
        if classification:
            total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
            correct += (logits.argmax(dim=1) == yb).sum().item()
            n += yb.size(0)
        else:
            pred = logits.reshape(-1)
            total_loss += F.mse_loss(pred, yb.reshape(-1), reduction="sum").item()
            n += yb.numel()
    if n == 0:
        return 0.0, 0.0
    acc = correct / n if classification else float("nan")
    return total_loss / n, acc


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
    groups = decay_param_groups(model, cfg.weight_decay)
    if cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(groups, lr=cfg.lr)
    elif cfg.optimizer == "adam":
        opt = torch.optim.Adam(groups, lr=cfg.lr)
    else:
        opt = torch.optim.SGD(groups, lr=cfg.lr, momentum=cfg.momentum)
    history = History()
    best_state = copy.deepcopy(model.state_dict())
    best_val = -float("inf")
    stale = 0

    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            if cfg.input_noise > 0:
                xb = xb + torch.randn_like(xb) * cfg.input_noise
            if cfg.adversarial_eps > 0:
                model.eval()
                adv = fgsm(model, xb, yb, cfg.adversarial_eps)
                model.train()
                opt.zero_grad(set_to_none=True)
                # mix clean + worst-case, not replace the whole batch
                mix = (torch.rand(len(xb), device=xb.device) < 0.5).view(
                    (-1,) + (1,) * (xb.ndim - 1)
                )
                xb = torch.where(mix, adv, xb)

            if cfg.activation_l1 > 0 and isinstance(model, MLP):
                logits, hidden = model(xb, return_hidden=True)
            else:
                logits = model(xb)
                hidden = None

            loss = F.cross_entropy(logits, yb, label_smoothing=cfg.label_smoothing)
            if cfg.l1 > 0:
                loss = loss + cfg.l1 * weight_l1(model)
            if cfg.activation_l1 > 0 and hidden is not None:
                # §7.10: penalty on activations, not weights
                loss = loss + cfg.activation_l1 * hidden.abs().mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        tr_loss, tr_acc = evaluate(model, train_loader)
        va_loss, va_acc = evaluate(model, val_loader)
        history.train_loss.append(tr_loss)
        history.train_acc.append(tr_acc)
        history.val_loss.append(va_loss)
        history.val_acc.append(va_acc)

        # classification: maximize val acc (val loss can dip before the net has learned)
        if va_acc > best_val + 1e-6:
            best_val = va_acc
            best_state = copy.deepcopy(model.state_dict())
            history.best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if (
                cfg.early_stop_patience is not None
                and epoch >= cfg.early_stop_min_epoch
                and stale >= cfg.early_stop_patience
            ):
                history.stopped_epoch = epoch
                break
        history.stopped_epoch = epoch

    if cfg.restore_best or cfg.early_stop_patience is not None:
        model.load_state_dict(best_state)
    return history


def metrics_for(
    model: torch.nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Metrics:
    tr = evaluate(model, _loader(X_train, y_train, 128, False))
    va = evaluate(model, _loader(X_val, y_val, 128, False))
    te = evaluate(model, _loader(X_test, y_test, 128, False))
    return Metrics(
        train_loss=tr[0],
        train_acc=tr[1],
        val_loss=va[0],
        val_acc=va[1],
        test_loss=te[0],
        test_acc=te[1],
        gap=tr[1] - te[1],
    )


@torch.no_grad()
def predict_proba(model: torch.nn.Module, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    loader = _loader(X, np.zeros(len(X), dtype=np.int64), batch_size, False)
    chunks = []
    for xb, _ in loader:
        chunks.append(F.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def bagged_proba(
    models: list[torch.nn.Module], X: np.ndarray
) -> np.ndarray:
    """§7.11: uniform average of bootstrap models."""
    return np.mean([predict_proba(m, X) for m in models], axis=0)


def bootstrap_indices(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=n)
