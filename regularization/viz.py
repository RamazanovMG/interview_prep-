"""Notebook plots. Each function is one concept — zeros, masks, attacks, etc."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso, LogisticRegression

from closed_form import l1_soft_threshold, l2_shrink_diag, l2_shrink_eigen
from models import fgsm

ZERO = 1e-4

try:
    from ipywidgets import FloatSlider, interact as _interact

    HAS_WIDGETS = True
except Exception:
    FloatSlider = None  # type: ignore[misc, assignment]
    _interact = None
    HAS_WIDGETS = False


def fslider(
    value: float,
    min: float,
    max: float,
    step: float,
    description: str,
    continuous_update: bool = True,
):
    """ipywidgets slider, or the default float if widgets aren't installed."""
    if HAS_WIDGETS:
        return FloatSlider(
            value=value,
            min=min,
            max=max,
            step=step,
            description=description,
            continuous_update=continuous_update,
            readout_format=".3g",
            style={"description_width": "initial"},
            layout={"width": "72%"},
        )
    return float(value)


def play(fn, **kwargs) -> None:
    """Bind sliders to `fn`. Falls back to one shot at the default values."""
    if HAS_WIDGETS:
        _interact(fn, **kwargs)
        return
    print("pip install ipywidgets  — then re-run setup for sliders")
    fn(**{k: getattr(v, "value", v) for k, v in kwargs.items()})


def l2_geometry(alpha: float = 0.55) -> None:
    """Fig 7.1: poorly determined axis gets smashed; well-determined axis barely moves."""
    w_star = np.array([1.8, 0.7])
    H = np.diag([0.15, 2.4])
    alpha = max(float(alpha), 1e-8)
    w_t = l2_shrink_eigen(w_star, H, alpha)
    scale = np.diag(H) / (np.diag(H) + alpha)
    print(f"w*      {w_star}")
    print(f"w_tilde {np.round(w_t, 3)}")
    print(f"λ/(λ+α) {np.round(scale, 3)}   (w1 poorly determined, w2 pinned)")

    w1 = np.linspace(-0.6, 2.4, 240)
    w2 = np.linspace(-1.2, 1.8, 240)
    W1, W2 = np.meshgrid(w1, w2)
    delta = np.stack([W1 - w_star[0], W2 - w_star[1]], axis=-1)
    J = 0.5 * np.einsum("...i,ij,...j->...", delta, H, delta)
    R = 0.5 * (W1**2 + W2**2)

    fig, ax = plt.subplots(figsize=(5.8, 5.2), layout="constrained")
    ax.contour(W1, W2, J, levels=8, colors="#1f77b4")
    ax.contour(W1, W2, R, levels=8, colors="#d62728", linestyles="--")
    ax.plot(*w_star, "o", color="#1f77b4", ms=9, label=r"$w^*$ unregularized")
    ax.plot(*w_t, "s", color="#d62728", ms=9, label=r"$\tilde{w}$ with $L_2$")
    ax.annotate("", xy=w_t, xytext=w_star, arrowprops=dict(arrowstyle="->", color="#333", lw=1.4))
    ax.axhline(0, color="#ccc", lw=0.6)
    ax.axvline(0, color="#ccc", lw=0.6)
    ax.set_xlabel(r"$w_1$ poorly determined ($\lambda=0.15$)")
    ax.set_ylabel(r"$w_2$ well determined ($\lambda=2.4$)")
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(rf"$L_2$ geometry, $\alpha={alpha:.3g}$")
    plt.show()


def zero_moment(alpha: float = 0.8, H: float = 1.0) -> None:
    """One α for both penalties. L1 snaps to 0 at α = |w*| H; L2 never does."""
    alpha = max(float(alpha), 1e-8)
    H = float(H)
    probes = np.array([0.3, 0.8, 1.5, 2.5])
    h = np.full_like(probes, H)
    l1_now = l1_soft_threshold(probes, h, alpha)
    l2_now = l2_shrink_diag(probes, h, alpha)
    kill_at = probes * H  # α where L1 hits 0

    print(f"{'w*':>6}  {'α_zero L1':>10}  {'L1 now':>8}  {'L2 now':>8}  dead?")
    for w, a0, t1, t2 in zip(probes, kill_at, l1_now, l2_now):
        dead = abs(t1) < 1e-12
        print(f"{w:6.2f}  {a0:10.2f}  {t1:8.3f}  {t2:8.3f}  {'YES ←' if dead else 'no'}")

    grid = np.linspace(-3.2, 3.2, 500)
    Hg = np.full_like(grid, H)
    thresh = alpha / H
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), layout="constrained")

    ax = axes[0]
    ax.axvspan(-thresh, thresh, color="#ff7f0e", alpha=0.18, label=rf"L1 dead $|w^*|<\alpha/H={thresh:.2f}$")
    ax.plot(grid, grid, color="#bbb", lw=1, label=r"$w^*$")
    ax.plot(grid, l2_shrink_diag(grid, Hg, alpha), color="#1f77b4", lw=2, label="L2")
    ax.plot(grid, l1_soft_threshold(grid, Hg, alpha), color="#ff7f0e", lw=2, label="L1")
    colors = ["#6a3d9a", "#33a02c", "#e31a1c", "#ff7f00"]
    for w, t1, t2, c in zip(probes, l1_now, l2_now, colors):
        ax.plot([w, w], [w, t2], color="#1f77b4", lw=0.8, ls=":")
        ax.plot([w, w], [w, t1], color="#ff7f0e", lw=0.8, ls=":")
        ax.scatter([w], [w], color=c, s=28, zorder=4)
        ax.scatter([w], [t2], color="#1f77b4", marker="s", s=36, zorder=5)
        ax.scatter([w], [t1], color="#ff7f0e", marker="x", s=50, zorder=6, linewidths=1.6)
    ax.axhline(0, color="#aaa", lw=0.6)
    ax.axvline(0, color="#aaa", lw=0.6)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$w^*$")
    ax.set_ylabel(r"$\tilde{w}(\alpha)$")
    ax.set_title(rf"same $\alpha={alpha:.2g}$: × = L1, square = L2")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[1]
    a_grid = np.linspace(0.0, 4.0, 400)
    for w, c, a0 in zip(probes, colors, kill_at):
        l1_path = l1_soft_threshold(np.full_like(a_grid, w), np.full_like(a_grid, H), a_grid)
        l2_path = l2_shrink_diag(np.full_like(a_grid, w), np.full_like(a_grid, H), np.maximum(a_grid, 1e-12))
        ax.plot(a_grid, l2_path, color=c, lw=1.6, ls="--")
        ax.plot(a_grid, l1_path, color=c, lw=2.0, label=rf"$w^*={w:g}$  (L1 dies at $\alpha={a0:g}$)")
        ax.scatter([alpha], [l1_soft_threshold(np.array([w]), np.array([H]), alpha)[0]], color=c, marker="x", s=50, zorder=5)
        ax.scatter([alpha], [l2_shrink_diag(np.array([w]), np.array([H]), alpha)[0]], color=c, marker="s", s=28, zorder=5)
    ax.axvline(alpha, color="#333", ls=":", lw=1.2)
    ax.axhline(0, color="#aaa", lw=0.6)
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.05, 2.7)
    ax.set_xlabel(r"$\alpha$  (same for L1 and L2)")
    ax.set_ylabel(r"$\tilde{w}(\alpha)$")
    ax.set_title("path: L1 — solid (hits 0), L2 — dashed (never)")
    ax.legend(frameon=False, fontsize=7)
    plt.show()


def shrinkage_1d(alpha_l2: float = 1.0, alpha_l1: float = 1.0) -> None:
    """L2 never hits 0; L1 is exactly 0 inside the orange dead zone."""
    w_star = np.linspace(-3, 3, 500)
    H = np.ones_like(w_star)
    alpha_l2 = max(float(alpha_l2), 1e-8)
    alpha_l1 = max(float(alpha_l1), 1e-8)
    thresh = alpha_l1 / H[0]
    l1 = l1_soft_threshold(w_star, H, alpha_l1)
    l2 = l2_shrink_diag(w_star, H, alpha_l2)
    probe = 0.3
    print(
        f"at w*={probe}:  L2 -> {float(l2_shrink_diag(np.array([probe]), np.array([1.0]), alpha_l2)[0]):.3f}"
        f"   L1 -> {float(l1_soft_threshold(np.array([probe]), np.array([1.0]), alpha_l1)[0]):.3f}"
        f"   (L1 dead zone |w*| < {thresh:.2f})"
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.4), layout="constrained")
    ax.axvspan(
        -thresh,
        thresh,
        color="#ff7f0e",
        alpha=0.18,
        label=rf"L1 dead zone $|w^*|<\alpha_1/H={thresh:.2f}$",
    )
    ax.plot(w_star, w_star, color="#bbb", label=r"$w^*$")
    ax.plot(w_star, l2, color="#1f77b4", lw=2, label=rf"$L_2$  $\alpha={alpha_l2:.2g}$")
    ax.plot(w_star, l1, color="#ff7f0e", lw=2, label=rf"$L_1$  $\alpha={alpha_l1:.2g}$")
    ax.scatter([-thresh, thresh], [0, 0], color="#d62728", zorder=5, s=40)
    ax.axhline(0, color="#aaa", lw=0.6)
    ax.axvline(0, color="#aaa", lw=0.6)
    ax.set_xlabel(r"$w^*$")
    ax.set_ylabel(r"$\tilde{w}$")
    ax.set_title("L1 hits 0 inside the band; L2 only shrinks")
    ax.legend(frameon=False, fontsize=8)
    plt.show()


def diabetes_stems(ridge_alpha: float = 2.0, lasso_alpha: float = 0.8) -> None:
    """Refit ridge/lasso. Red × = exact zeros (L1 only)."""
    from sklearn.linear_model import Lasso, LinearRegression, Ridge

    from data import load_diabetes_split

    d = load_diabetes_split()
    ols = LinearRegression().fit(d.X_train, d.y_train)
    ridge = Ridge(alpha=float(ridge_alpha)).fit(d.X_train, d.y_train)
    lasso = Lasso(alpha=max(float(lasso_alpha), 1e-8), max_iter=20000).fit(d.X_train, d.y_train)

    def mse(m) -> float:
        return float(np.mean((m.predict(d.X_test) - d.y_test) ** 2))

    for name, m in [("OLS", ols), ("Ridge", ridge), ("Lasso", lasso)]:
        z = int(np.sum(np.abs(m.coef_) < 1e-6))
        dead = [d.feature_names[i] for i, v in enumerate(m.coef_) if abs(v) < 1e-6]
        print(f"{name:6}  test MSE={mse(m):8.1f}  zeros={z}/{m.coef_.size}  dropped={dead}")

    stems_with_zeros(
        {
            "OLS": ols.coef_,
            f"Ridge α={ridge_alpha:g}": ridge.coef_,
            f"Lasso α={lasso_alpha:g}": lasso.coef_,
        },
        d.feature_names,
        title="diabetes — red × are exact zeros (L1)",
    )


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
