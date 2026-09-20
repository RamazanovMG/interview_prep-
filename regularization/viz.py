"""Notebook plots. Each function is one concept — zeros, masks, attacks, etc."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso, LogisticRegression

from models import fgsm

ZERO = 1e-4


def flatten_weights(model: torch.nn.Module) -> np.ndarray:
    chunks = [p.detach().cpu().ravel() for p in model.parameters() if p.ndim > 1]
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return torch.cat(chunks).numpy()


def stems_with_zeros(
    estimates: dict[str, np.ndarray],
    feature_names: tuple[str, ...] | list[str] | None = None,
    zero_tol: float = ZERO,
    title: str = "",
) -> None:
    """Nonzeros as stems; exact zeros as red ×. That's the L1 punchline."""
    first = np.asarray(next(iter(estimates.values()))).ravel()
    n = len(first)
    fig, axes = plt.subplots(
        len(estimates), 1, figsize=(9.2, 2.15 * len(estimates)), sharex=True, layout="constrained"
    )
    if len(estimates) == 1:
        axes = [axes]
    palette = ["#4c4c4c", "#1f77b4", "#ff7f0e", "#2ca02c"]
    x = np.arange(n)
    for ax, (name, w), c in zip(axes, estimates.items(), palette):
        w = np.asarray(w).ravel()
        dead = np.abs(w) < zero_tol
        ax.axhline(0, color="#bbb", lw=0.6)
        if np.any(~dead):
            ax.stem(x[~dead], w[~dead], linefmt=c, markerfmt="o", basefmt="none")
        if np.any(dead):
            ax.scatter(
                x[dead],
                np.zeros(int(dead.sum())),
                marker="x",
                s=55,
                color="#d62728",
                linewidths=1.6,
                zorder=5,
                label=f"{int(dead.sum())} zeros",
            )
            ax.legend(loc="upper right", frameon=False, fontsize=8)
        ax.set_ylabel(name)
        ax.set_title(f"{int(dead.sum())}/{n} exact zeros", loc="left", fontsize=10)
    if feature_names is not None and len(feature_names) == n:
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(list(feature_names), rotation=40, ha="right")
    else:
        axes[-1].set_xlabel("weight index")
    if title:
        fig.suptitle(title, y=1.02)
    plt.show()


def lasso_zero_path(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: tuple[str, ...] | list[str],
    alphas: np.ndarray | None = None,
) -> None:
    """Sweep λ: more L1 → more coefficients sit on exactly 0."""
    if alphas is None:
        alphas = np.logspace(-2, 1.0, 14)
    n_zeros, mse, paths = [], [], []
    for a in alphas:
        m = Lasso(alpha=float(a), max_iter=20000).fit(X_train, y_train)
        n_zeros.append(int(np.sum(np.abs(m.coef_) < ZERO)))
        mse.append(float(np.mean((m.predict(X_test) - y_test) ** 2)))
        paths.append(m.coef_.copy())
    paths = np.stack(paths)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), layout="constrained")
    ax = axes[0]
    for j, name in enumerate(feature_names):
        ax.plot(alphas, paths[:, j], lw=1.5, label=name)
    ax.axhline(0, color="#aaa", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel(r"lasso $\alpha$")
    ax.set_ylabel("coefficient")
    ax.set_title("coefficients collapsing onto 0")
    ax.legend(ncol=2, fontsize=7, frameon=False)

    ax = axes[1]
    ax.plot(alphas, n_zeros, color="#d62728", marker="o", label="# zeros")
    ax.set_xscale("log")
    ax.set_xlabel(r"lasso $\alpha$")
    ax.set_ylabel("# exact zeros", color="#d62728")
    ax2 = ax.twinx()
    ax2.plot(alphas, mse, color="#1f77b4", marker="s", label="test MSE")
    ax2.set_ylabel("test MSE", color="#1f77b4")
    ax.set_title("sparsity vs fit")
    plt.show()


def logreg_zero_path(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: tuple[str, ...] | list[str],
    Cs: np.ndarray | None = None,
) -> None:
    """Smaller C = more L1. Watch features drop out."""
    if Cs is None:
        Cs = np.logspace(-1.2, 1.2, 10)
    n_zeros, paths = [], []
    for C in Cs:
        m = LogisticRegression(
            C=float(C), l1_ratio=1.0, solver="saga", max_iter=8000, random_state=0
        ).fit(X_train, y_train)
        w = m.coef_.ravel()
        n_zeros.append(int(np.sum(np.abs(w) < ZERO)))
        paths.append(w.copy())
    paths = np.stack(paths)
    keep = np.max(np.abs(paths), axis=0) > ZERO

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), layout="constrained")
    ax = axes[0]
    for j, name in enumerate(feature_names):
        if not keep[j]:
            continue
        ax.plot(Cs, paths[:, j], lw=1.4, label=name)
    ax.axhline(0, color="#aaa", lw=0.6)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("C  (← more L1)")
    ax.set_ylabel("weight")
    ax.set_title("L1 logreg path (nonzero-at-some-C only)")
    ax.legend(fontsize=6, ncol=2, frameon=False)

    ax = axes[1]
    ax.plot(Cs, n_zeros, color="#d62728", marker="o")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("C  (← more L1)")
    ax.set_ylabel("# exact zeros")
    ax.set_ylim(0, len(feature_names) + 1)
    ax.set_title(f"{len(feature_names)} features")
    plt.show()


def weight_sparsity(
    models: dict[str, torch.nn.Module],
    near: float = 1e-3,
    title: str = "weight magnitude",
) -> None:
    """L1 piles mass at 0; L2 shrinks the tail; dropout does neither."""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), layout="constrained")
    bins = np.logspace(-6, 1.2, 50)
    for name, m in models.items():
        w = np.abs(flatten_weights(m)) + 1e-12
        axes[0].hist(w, bins=bins, density=True, histtype="step", lw=1.8, label=name)
        print(
            f"{name:18} median|w|={np.median(w):.4f}  "
            f"frac<|{near}|={np.mean(w < near):.3f}  "
            f"l0@1e-4={int(np.sum(w < 1e-4))}/{w.size}"
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$|w|$")
    axes[0].set_ylabel("density")
    axes[0].set_title("histogram")
    axes[0].legend(frameon=False)

    th = np.logspace(-6, -1, 40)
    for name, m in models.items():
        w = np.abs(flatten_weights(m))
        axes[1].plot(th, [np.mean(w < t) for t in th], lw=2, label=name)
    axes[1].axvline(near, color="#aaa", ls=":", lw=1)
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"threshold $t$")
    axes[1].set_ylabel(r"fraction $|w|<t$")
    axes[1].set_title("sparsity CDF  (L1 jumps)")
    axes[1].legend(frameon=False)
    fig.suptitle(title, y=1.03)
    plt.show()


def first_layer_abs(model: torch.nn.Module, title: str = "first layer |W|") -> None:
    W = None
    for p in model.parameters():
        if p.ndim == 2:
            W = p.detach().cpu().abs().numpy()
            break
    if W is None:
        print("no dense weight")
        return
    # 256 x 784 is unreadable; show a slice + a zero-mask
    sl = W[:64, :96]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), layout="constrained")
    im = axes[0].imshow(sl, cmap="magma", aspect="auto")
    axes[0].set_title("|W|  (64 units × 96 pixels)")
    axes[0].set_xlabel("input dim")
    axes[0].set_ylabel("hidden unit")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    axes[1].imshow(sl < 1e-3, cmap="gray_r", aspect="auto", vmin=0, vmax=1)
    axes[1].set_title("white = |W| < 1e-3  (L1 punches holes)")
    fig.suptitle(title, y=1.03)
    plt.show()


def dropout_passes(
    model: torch.nn.Module,
    x: np.ndarray,
    n_passes: int = 10,
    n_units: int = 48,
    title: str = "dropout mask across stochastic passes",
) -> None:
    """Rows = independent train-mode forwards. Black = dropped unit. Eval is dense."""
    xt = torch.from_numpy(np.ascontiguousarray(x[:1]))
    model.train()
    rows = []
    with torch.no_grad():
        for _ in range(n_passes):
            _, h = model(xt, return_hidden=True)
            rows.append(h[0, :n_units].cpu().numpy())
        model.eval()
        _, h_eval = model(xt, return_hidden=True)
        ev = h_eval[0, :n_units].cpu().numpy()
    train_h = np.stack(rows)
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 4.4), sharex=True, layout="constrained")
    axes[0].imshow(np.abs(train_h) > 1e-8, cmap="gray_r", aspect="auto", interpolation="nearest")
    axes[0].set_ylabel("train pass")
    axes[0].set_title("white = survived  /  black = dropped")
    axes[1].imshow(ev[None, :], cmap="coolwarm", aspect="auto", interpolation="nearest")
    axes[1].set_yticks([0])
    axes[1].set_yticklabels(["eval"])
    axes[1].set_xlabel("hidden unit")
    fig.suptitle(title, y=1.03)
    plt.show()


def noise_grid(X: np.ndarray, noise: float, n: int = 6, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    imgs = X[:n]
    noisy = np.clip(imgs + rng.normal(0, noise, imgs.shape).astype(imgs.dtype), 0, 1)
    _pair_rows(imgs, noisy, f"clean", f"noise σ={noise}")


def shift_grid(X: np.ndarray, shifted: np.ndarray, n: int = 8) -> None:
    _pair_rows(X[:n], shifted[:n], "original", "shifted (aug)")


def _pair_rows(a: np.ndarray, b: np.ndarray, ta: str, tb: str) -> None:
    n = len(a)
    fig, axes = plt.subplots(2, n, figsize=(1.45 * n, 3.2), layout="constrained")
    for i in range(n):
        for row, src in enumerate((a, b)):
            img = src[i]
            if img.ndim == 4:
                img = img[0]
            if img.ndim == 3:
                img = img[0] if img.shape[0] in (1, 3) else img.squeeze()
            if img.ndim == 1:
                s = int(np.sqrt(img.size))
                img = img.reshape(s, s)
            axes[row, i].imshow(img, cmap="gray")
            axes[row, i].axis("off")
    axes[0, 0].set_ylabel(ta)
    axes[1, 0].set_ylabel(tb)
    for row in range(2):
        axes[row, 0].axis("on")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        for spine in axes[row, 0].spines.values():
            spine.set_visible(False)
    plt.show()


def confidence_hist(
    models: dict[str, torch.nn.Module],
    X: np.ndarray,
    title: str = "max softmax  (label smoothing clips overconfidence)",
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
    xt = torch.from_numpy(np.ascontiguousarray(X))
    for name, m in models.items():
        m.eval()
        with torch.no_grad():
            p = F.softmax(m(xt), dim=1).max(dim=1).values.cpu().numpy()
        ax.hist(p, bins=25, range=(0.2, 1.0), density=True, histtype="step", lw=1.8, label=name)
        print(f"{name:18} mean max-p={p.mean():.3f}  frac>0.95={np.mean(p > 0.95):.3f}")
    ax.set_xlabel("max softmax")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(frameon=False)
    plt.show()


def overlay_val(hists: dict, title: str = "val acc") -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
    for name, h in hists.items():
        ax.plot(h.val_acc, lw=2, label=f"{name} val")
        ax.plot(h.train_acc, lw=1, ls="--", alpha=0.6, label=f"{name} train")
        if getattr(h, "best_epoch", None) is not None and "early" in name:
            ax.axvline(h.best_epoch, color="#333", ls=":", lw=1)
            ax.scatter([h.best_epoch], [h.val_acc[h.best_epoch]], color="#d62728", zorder=5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    plt.show()


def conv_kernels(model: torch.nn.Module, title: str = "shared 3×3 kernels (ch. 7.9)") -> None:
    W = None
    for p in model.parameters():
        if p.ndim == 4:
            W = p.detach().cpu().numpy()
            break
    if W is None:
        print("no conv weight")
        return
    n = min(16, W.shape[0])
    fig, axes = plt.subplots(2, 8, figsize=(8.8, 2.6), layout="constrained")
    vmax = np.percentile(np.abs(W[:n]), 99)
    for ax, k in zip(axes.ravel(), W[:n]):
        ax.imshow(k[0], cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, y=1.04)
    plt.show()


def hidden_sparsity(h0: np.ndarray, h1: np.ndarray, name0: str, name1: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), layout="constrained")
    axes[0].hist(h0.ravel(), bins=60, density=True, alpha=0.55, label=name0)
    axes[0].hist(h1.ravel(), bins=60, density=True, alpha=0.55, label=name1)
    axes[0].set_xlabel("h")
    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False, fontsize=8)
    n = min(48, h0.shape[0])
    u = min(64, h0.shape[1])
    axes[1].imshow(np.abs(h0[:n, :u]), cmap="magma", aspect="auto")
    axes[1].set_title(name0)
    axes[1].set_xlabel("unit")
    axes[1].set_ylabel("sample")
    axes[2].imshow(np.abs(h1[:n, :u]), cmap="magma", aspect="auto")
    axes[2].set_title(name1)
    plt.show()


def fgsm_triplets(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    eps: float,
    labels: tuple[str, ...] | None = None,
    n: int = 6,
) -> None:
    model.eval()
    xt = torch.from_numpy(np.ascontiguousarray(X[:n]))
    yt = torch.from_numpy(np.ascontiguousarray(y[:n])).long()
    adv = fgsm(model, xt, yt, eps)
    with torch.no_grad():
        pred_c = model(xt).argmax(1).cpu().numpy()
        pred_a = model(adv).argmax(1).cpu().numpy()
    clean = xt.detach().cpu().numpy()
    atk = adv.cpu().numpy()
    delta = atk - clean
    fig, axes = plt.subplots(3, n, figsize=(1.55 * n, 5.0), layout="constrained")
    for i in range(n):
        for row, img in enumerate((clean[i], delta[i], atk[i])):
            view = img
            if view.ndim == 3:
                view = view[0]
            if view.ndim == 1:
                s = int(np.sqrt(view.size))
                view = view.reshape(s, s)
            if row == 1:
                axes[row, i].imshow(view, cmap="coolwarm", vmin=-eps, vmax=eps)
            else:
                axes[row, i].imshow(view, cmap="gray")
            axes[row, i].axis("off")
        lab = labels[int(y[i])] if labels is not None else str(int(y[i]))
        pc = labels[int(pred_c[i])] if labels is not None else str(int(pred_c[i]))
        pa = labels[int(pred_a[i])] if labels is not None else str(int(pred_a[i]))
        axes[0, i].set_title(f"{lab}\n→ {pc}", fontsize=7, color="#1a1a1a")
        axes[2, i].set_title(f"→ {pa}", fontsize=7, color="#d62728" if pred_a[i] != y[i] else "#2ca02c")
    axes[0, 0].set_ylabel("clean")
    axes[1, 0].set_ylabel("sign(∇xJ)")
    axes[2, 0].set_ylabel("adv")
    fig.suptitle(rf"FGSM $\varepsilon={eps}$  (red pred = flipped)", y=1.03)
    plt.show()


def bag_bars(member_test: list[float], bag_test: float, bag_train: float) -> None:
    names = [f"m{i}" for i in range(len(member_test))] + ["bag"]
    vals = member_test + [bag_test]
    fig, ax = plt.subplots(figsize=(max(5.5, 0.7 * len(names)), 3.6), layout="constrained")
    colors = ["#9ecae1"] * len(member_test) + ["#2171b5"]
    ax.bar(names, vals, color=colors)
    ax.axhline(bag_train, color="#d62728", ls=":", label=f"bag train {bag_train:.3f}")
    ax.set_ylim(0.3, 1.02)
    ax.set_ylabel("test acc")
    ax.set_title("bagging: average of bootstrap members")
    ax.legend(frameon=False)
    plt.show()


def param_bars(counts: dict[str, int]) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.4), layout="constrained")
    ax.bar(list(counts), list(counts.values()), color=["#ff7f0e", "#1f77b4"][: len(counts)])
    ax.set_ylabel("# parameters")
    ax.set_title("sharing ⇒ fewer unique weights")
    for i, v in enumerate(counts.values()):
        ax.text(i, v, f" {v:,}", va="bottom", ha="center", fontsize=9)
    plt.show()


def scoreboard(board: dict[str, dict]) -> None:
    if not board:
        print("board empty — train something first")
        return
    print(f"{'name':18} {'train':>7} {'val':>7} {'test':>7} {'gap':>7}")
    for k, m in board.items():
        print(f"{k:18} {m['train']:7.3f} {m['val']:7.3f} {m['test']:7.3f} {m['gap']:7.3f}")
    names = list(board)
    fig, ax = plt.subplots(figsize=(max(6.5, 0.7 * len(names)), 4.0), layout="constrained")
    x = np.arange(len(names))
    ax.bar(x - 0.18, [board[k]["train"] for k in names], 0.36, label="train", color="#9ecae1")
    ax.bar(x + 0.18, [board[k]["test"] for k in names], 0.36, label="test", color="#2171b5")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0.4, 1.03)
    ax.set_ylabel("accuracy")
    ax.legend(frameon=False)
    plt.show()
