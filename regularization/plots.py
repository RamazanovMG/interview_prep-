"""Matplotlib helpers. Headless Agg backend."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from closed_form import l1_soft_threshold, l2_shrink_diag, l2_shrink_eigen
from train import predict_proba

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

PALETTE = {
    "none": "#4c4c4c",
    "l2": "#1f77b4",
    "l1": "#ff7f0e",
    "dropout": "#2ca02c",
    "input_noise": "#9467bd",
    "early_stop": "#d62728",
    "label_smooth": "#8c564b",
    "l2+dropout": "#17becf",
    "cnn": "#1f77b4",
    "mlp": "#ff7f0e",
    "bagging": "#2ca02c",
    "adv": "#e377c2",
}


def save(fig: plt.Figure, name: str) -> Path:
    path = RESULTS / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_fashion_grid(
    X: np.ndarray,
    y: np.ndarray,
    labels: tuple[str, ...],
    path: str = "fashion_samples.png",
    n: int = 40,
) -> Path:
    """Sanity check: these are real Zalando images, not generated blobs."""
    imgs = X.reshape(len(X), -1)
    side = int(np.sqrt(imgs.shape[1]))
    fig, axes = plt.subplots(4, 10, figsize=(11, 4.6), layout="constrained")
    for ax, img, lab in zip(axes.ravel(), imgs[:n], y[:n]):
        ax.imshow(img.reshape(side, side), cmap="gray")
        ax.set_title(labels[int(lab)], fontsize=7)
        ax.axis("off")
    fig.suptitle("Fashion-MNIST (Zalando) — train subset")
    return save(fig, path)


def plot_l2_geometry(path: str = "fig71_l2_geometry.png") -> Path:
    """Recreate the geometry of Fig. 7.1: elongated loss vs L2 circles."""
    w_star = np.array([1.8, 0.7])
    # small λ along w1 → regularizer dominates that axis
    H = np.diag([0.15, 2.4])
    alpha = 0.55
    w_tilde = l2_shrink_eigen(w_star, H, alpha)

    w1 = np.linspace(-0.6, 2.4, 240)
    w2 = np.linspace(-1.2, 1.8, 240)
    W1, W2 = np.meshgrid(w1, w2)
    delta = np.stack([W1 - w_star[0], W2 - w_star[1]], axis=-1)
    J = 0.5 * np.einsum("...i,ij,...j->...", delta, H, delta)
    R = 0.5 * (W1**2 + W2**2)

    fig, ax = plt.subplots(figsize=(6.2, 5.4), layout="constrained")
    ax.contour(W1, W2, J, levels=8, colors="#1f77b4", linewidths=1.2)
    ax.contour(W1, W2, R, levels=8, colors="#d62728", linestyles="--", linewidths=0.9)
    ax.plot(*w_star, "o", color="#1f77b4", ms=8, label=r"$w^*$ unregularized")
    ax.plot(*w_tilde, "s", color="#d62728", ms=8, label=r"$\tilde w$ with $L_2$")
    ax.annotate(
        r"small $\lambda_1$: pulled toward 0",
        xy=tuple(w_tilde),
        xytext=(1.15, 1.45),
        arrowprops=dict(arrowstyle="->", color="#444"),
        fontsize=9,
    )
    ax.axhline(0, color="#aaa", lw=0.6)
    ax.axvline(0, color="#aaa", lw=0.6)
    ax.set_xlabel(r"$w_1$  (poorly determined)")
    ax.set_ylabel(r"$w_2$  (well determined)")
    ax.set_title(r"Fig 7.1: $L_2$ shrinks small-$\lambda$ directions")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", frameon=False)
    return save(fig, path)


def plot_constraint_sets(path: str = "l1_l2_constraint_sets.png") -> Path:
    """§7.2: penalty ≡ constrained opt. L1 diamond hits axes → sparsity."""
    theta = np.linspace(0, 2 * np.pi, 400)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), layout="constrained")

    # L2 ball
    axes[0].plot(np.cos(theta), np.sin(theta), color="#1f77b4", lw=2)
    axes[0].plot([0.72], [0.69], "o", color="#d62728")
    axes[0].set_title(r"$L_2$ ball $\|w\|_2 \leq k$")

    # L1 diamond
    diamond = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]], dtype=float)
    axes[1].plot(diamond[:, 0], diamond[:, 1], color="#ff7f0e", lw=2)
    axes[1].plot([0.0], [1.0], "o", color="#d62728")
    axes[1].set_title(r"$L_1$ diamond $\|w\|_1 \leq k$  (hits axis)")

    for ax in axes:
        # loss ellipses centered off-axis so L1 optimum is a corner
        w1 = np.linspace(-1.4, 1.4, 200)
        w2 = np.linspace(-1.4, 1.4, 200)
        W1, W2 = np.meshgrid(w1, w2)
        J = 0.5 * ((W1 - 1.15) ** 2 / 0.7 + (W2 - 1.05) ** 2 / 0.55)
        ax.contour(W1, W2, J, levels=6, colors="#888", linewidths=0.8)
        ax.axhline(0, color="#ccc", lw=0.6)
        ax.axvline(0, color="#ccc", lw=0.6)
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$w_1$")
        ax.set_ylabel(r"$w_2$")
    return save(fig, path)


def plot_shrinkage_1d(path: str = "l1_vs_l2_shrinkage.png") -> Path:
    """Eq. 7.13 vs 7.23 on a 1-D slice: L1 soft-threshold, L2 always-nonzero."""
    w_star = np.linspace(-3, 3, 400)
    H = np.full_like(w_star, 1.0)
    alpha = 1.0
    fig, ax = plt.subplots(figsize=(6.4, 4.4), layout="constrained")
    ax.plot(w_star, w_star, color="#bbbbbb", lw=1, label=r"$w^*$")
    ax.plot(w_star, l2_shrink_diag(w_star, H, alpha), color="#1f77b4", lw=2, label=r"$L_2$  $\lambda/(\lambda+\alpha)$")
    ax.plot(w_star, l1_soft_threshold(w_star, H, alpha), color="#ff7f0e", lw=2, label=r"$L_1$  soft-threshold")
    ax.axhline(0, color="#aaa", lw=0.6)
    ax.axvline(0, color="#aaa", lw=0.6)
    ax.set_xlabel(r"$w^*$")
    ax.set_ylabel(r"$\tilde w$")
    ax.set_title(r"$L_1$ hits 0; $L_2$ only shrinks")
    ax.legend(frameon=False)
    return save(fig, path)


def plot_weight_stems(
    estimates: dict[str, np.ndarray],
    path: str = "linear_weights.png",
    feature_names: tuple[str, ...] | None = None,
) -> Path:
    first = next(iter(estimates.values()))
    idx = np.arange(len(first))
    fig, axes = plt.subplots(
        len(estimates),
        1,
        figsize=(9.5, 2.2 * len(estimates)),
        sharex=True,
        layout="constrained",
    )
    if len(estimates) == 1:
        axes = [axes]
    colors = ["#4c4c4c", "#1f77b4", "#ff7f0e", "#2ca02c"]
    for ax, (name, w), c in zip(axes, estimates.items(), colors):
        ax.stem(idx, w.ravel(), linefmt=c, markerfmt="o", basefmt="k-")
        n_zero = int(np.sum(np.abs(w) < 1e-4))
        ax.set_ylabel(name)
        pad = 0.2 + 0.05 * float(np.max(np.abs(w)))
        ax.set_ylim(w.min() - pad, w.max() + pad)
        ax.set_title(f"{name}  |  {n_zero}/{w.size} exact zeros", loc="left", fontsize=10)
    if feature_names and len(feature_names) == len(first):
        axes[-1].set_xticks(idx)
        axes[-1].set_xticklabels(feature_names, rotation=40, ha="right", fontsize=8)
    else:
        axes[-1].set_xlabel("feature")
    return save(fig, path)


def plot_decision_boundaries(
    models: dict[str, torch.nn.Module],
    X: np.ndarray,
    y: np.ndarray,
    path: str = "mlp_boundaries.png",
    flip: np.ndarray | None = None,
    pca=None,
) -> Path:
    names = list(models)
    cols = min(4, len(names))
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.3 * rows), layout="constrained")
    axes = np.atleast_1d(axes).ravel()
    if pca is not None:
        pts = pca.transform(X)
    else:
        pts = X[:, :2]
    pad = 0.6
    xx, yy = np.meshgrid(
        np.linspace(pts[:, 0].min() - pad, pts[:, 0].max() + pad, 180),
        np.linspace(pts[:, 1].min() - pad, pts[:, 1].max() + pad, 180),
    )
    grid2 = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    grid = pca.inverse_transform(grid2).astype(np.float32) if pca is not None else grid2
    for ax, name in zip(axes, names):
        model = models[name]
        proba = predict_proba(model, grid)
        if proba.shape[1] == 2:
            Z = proba[:, 1].reshape(xx.shape)
            ax.contourf(xx, yy, Z, levels=20, cmap="RdBu", vmin=0, vmax=1, alpha=0.85)
            ax.scatter(pts[:, 0], pts[:, 1], c=y, cmap="RdBu", s=22, edgecolors="k", linewidths=0.4)
        else:
            Z = proba.argmax(1).reshape(xx.shape)
            ax.contourf(xx, yy, Z, levels=np.arange(proba.shape[1] + 1) - 0.5, cmap="tab10", alpha=0.75)
            ax.scatter(pts[:, 0], pts[:, 1], c=y, cmap="tab10", s=22, edgecolors="k", linewidths=0.4)
        if flip is not None and flip.any():
            ax.scatter(
                pts[flip, 0],
                pts[flip, 1],
                s=55,
                marker="x",
                c="k",
                linewidths=1.2,
            )
        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(names) :]:
        ax.axis("off")
    fig.suptitle("Same MLP on sklearn digits — PCA plane slice", fontsize=12)
    return save(fig, path)


def plot_learning_curves(histories: dict, path: str = "learning_curves.png") -> Path:
    names = list(histories)
    cols = 2
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8.8, 3.3 * rows), layout="constrained")
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, names):
        hist = histories[name]
        epochs = np.arange(len(hist.train_loss))
        ax.plot(epochs, hist.train_loss, color="#1f77b4", lw=1.7, label="train")
        ax.plot(epochs, hist.val_loss, color="#d62728", lw=1.7, label="val")
        if hist.best_epoch:
            ax.axvline(hist.best_epoch, color="#333", ls=":", lw=1)
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(frameon=False, fontsize=8)
    for ax in axes[len(names) :]:
        ax.axis("off")
    fig.suptitle("Early stopping restores best val acc (ch. 7.8)")
    return save(fig, path)


def plot_comparison_bars(
    rows: list[dict],
    path: str = "comparison_bars.png",
    title: str = "Train vs test acc (gap = overfit)",
    ymin: float = 0.4,
) -> Path:
    names = [r["name"] for r in rows]
    train = [r["train_acc"] for r in rows]
    test = [r["test_acc"] for r in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(7.5, 0.9 * len(names)), 4.4), layout="constrained")
    ax.bar(x - 0.18, train, 0.36, label="train", color="#9ecae1")
    ax.bar(x + 0.18, test, 0.36, label="test", color="#2171b5")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(ymin, 1.03)
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.legend(frameon=False)
    for i, r in enumerate(rows):
        ax.text(i, ymin + 0.02, f"gap {r['gap']:.2f}", ha="center", fontsize=8, color="#333")
    return save(fig, path)


def plot_adversarial(rows: list[dict], path: str = "adversarial.png") -> Path:
    names = [r["name"] for r in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.6, 4.3), layout="constrained")
    ax.bar(x - 0.18, [r["clean_test_acc"] for r in rows], 0.36, label="clean test", color="#9ecae1")
    ax.bar(x + 0.18, [r["fgsm_test_acc"] for r in rows], 0.36, label="FGSM test", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Ch. 7.13: random-looking FGSM wrecks a clean net")
    ax.legend(frameon=False)
    return save(fig, path)


def plot_linear_mse(rows: list[dict], path: str = "linear_mse.png") -> Path:
    names = [r["name"] for r in rows]
    mse = [r["test_mse"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.8, 4.0), layout="constrained")
    ax.bar(names, mse, color=["#4c4c4c", "#1f77b4", "#ff7f0e"])
    ax.set_yscale("log")
    ax.set_ylabel("test MSE (log)")
    ax.set_title("sklearn diabetes: test MSE")
    return save(fig, path)


def plot_sparsity_hist(hidden_none: np.ndarray, hidden_l1: np.ndarray, path: str = "activation_sparsity.png") -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), layout="constrained")
    ax.hist(hidden_none.ravel(), bins=60, density=True, alpha=0.55, label="tanh, no act. penalty", color="#4c4c4c")
    ax.hist(hidden_l1.ravel(), bins=60, density=True, alpha=0.55, label="tanh + L1 on h", color="#ff7f0e")
    ax.set_xlabel("hidden activation")
    ax.set_ylabel("density")
    ax.set_title("Ch. 7.10: representational sparsity (penalty on h, not w)")
    ax.legend(frameon=False)
    return save(fig, path)
