"""Notebook plots. Each function is one ch. 8 concept."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from algos import (
    cliff_J,
    cliff_grad,
    newton_mse,
    quadratic_J,
    quadratic_grad,
    rotate_hessian,
    run_first_order,
    saddle_J,
    saddle_grad,
    safe_lr,
    sgd_mse,
)

try:
    from ipywidgets import FloatSlider, IntSlider, interact as _interact

    HAS_WIDGETS = True
except Exception:
    FloatSlider = None  # type: ignore[misc, assignment]
    IntSlider = None  # type: ignore[misc, assignment]
    _interact = None
    HAS_WIDGETS = False

W_STAR = np.array([1.4, 0.6])
COLORS = {
    "sgd": "#4c4c4c",
    "momentum": "#1f77b4",
    "nesterov": "#2ca02c",
    "adagrad": "#ff7f0e",
    "rmsprop": "#9467bd",
    "adam": "#d62728",
    "newton": "#8c564b",
}


def fslider(value, min, max, step, description, continuous_update=True):
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


def islider(value, min, max, step, description, continuous_update=False):
    if HAS_WIDGETS:
        return IntSlider(
            value=value,
            min=min,
            max=max,
            step=step,
            description=description,
            continuous_update=continuous_update,
            style={"description_width": "initial"},
            layout={"width": "72%"},
        )
    return int(value)


def play(fn, **kwargs) -> None:
    if HAS_WIDGETS:
        _interact(fn, **kwargs)
        return
    print("pip install ipywidgets  — then re-run setup for sliders")
    fn(**{k: getattr(v, "value", v) for k, v in kwargs.items()})


def _contour(ax, Jfn, xlim, ylim, levels=14):
    xs = np.linspace(*xlim, 220)
    ys = np.linspace(*ylim, 220)
    X, Y = np.meshgrid(xs, ys)
    Z = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = Jfn(np.array([X[i, j], Y[i, j]]))
    ax.contour(X, Y, Z, levels=levels, colors="#1f77b4", linewidths=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def _draw_path(ax, path, color, label):
    ax.plot(path[:, 0], path[:, 1], "-", color=color, lw=1.6, label=label)
    ax.plot(path[0, 0], path[0, 1], "o", color=color, ms=5)
    ax.plot(path[-1, 0], path[-1, 1], "s", color=color, ms=6)


def ill_conditioned(
    kappa: float = 25.0, lr: float = 0.08, steps: float = 40.0
) -> None:
    """Fig 8.2 vibe: skinny bowl, SGD zigzags along the long axis."""
    kappa = max(float(kappa), 1.0)
    lmax, lmin = 2.4, 2.4 / kappa
    hess = rotate_hessian(lmin, lmax, deg=35.0)
    cap = safe_lr(lmax)
    w0 = np.array([-1.1, 1.7])

    def g(w):
        return quadratic_grad(w, hess, W_STAR)

    path = run_first_order(w0, g, "sgd", float(lr), int(steps))
    end_J = quadratic_J(path[-1], hess, W_STAR)
    print(
        f"κ = λmax/λmin = {kappa:.1f}   λ = ({lmax:.2f}, {lmin:.3f})   "
        f"safe ε < {cap:.3f}   used ε = {lr:.3f}   J_end = {end_J:.4f}"
    )
    if lr >= cap:
        print("ε ≥ 2/λmax → SGD расходится на этой квадратике")

    fig, ax = plt.subplots(figsize=(5.6, 5.2), layout="constrained")
    _contour(ax, lambda w: quadratic_J(w, hess, W_STAR), (-2.0, 2.6), (-1.6, 2.4))
    _draw_path(ax, path, COLORS["sgd"], "SGD")
    ax.plot(*W_STAR, "*", color="#d62728", ms=12, label="min")
    ax.set_title("ill-conditioned quadratic")
    ax.legend(loc="upper right", fontsize=8)
    plt.show()


def first_order_bowl(
    lr: float = 0.08,
    beta: float = 0.85,
    kappa: float = 18.0,
    steps: float = 50.0,
) -> None:
    """SGD vs momentum vs Nesterov vs Adam on the same skinny bowl."""
    kappa = max(float(kappa), 1.0)
    lmax, lmin = 2.4, 2.4 / kappa
    hess = rotate_hessian(lmin, lmax, deg=35.0)
    w0 = np.array([-1.1, 1.7])

    def g(w):
        return quadratic_grad(w, hess, W_STAR)

    methods = ("sgd", "momentum", "nesterov", "adam")
    fig, ax = plt.subplots(figsize=(5.8, 5.2), layout="constrained")
    _contour(ax, lambda w: quadratic_J(w, hess, W_STAR), (-2.0, 2.6), (-1.6, 2.4))
    ax.plot(*W_STAR, "*", color="#d62728", ms=11)
    print(f"{'method':10}  J_end   ‖w−w*‖")
    for name in methods:
        path = run_first_order(
            w0, g, name, float(lr), int(steps), beta=float(beta)
        )
        _draw_path(ax, path, COLORS[name], name)
        j = quadratic_J(path[-1], hess, W_STAR)
        dist = float(np.linalg.norm(path[-1] - W_STAR))
        print(f"{name:10}  {j:6.4f}  {dist:7.3f}")
    ax.set_title("first-order methods, same bowl / same ε")
    ax.legend(loc="upper right", fontsize=8)
    plt.show()


def saddle_escape(lr: float = 0.15, bump: float = 0.04, steps: float = 40.0) -> None:
    """J = (x² − y²)/2. Along x it's a min, along y a max. SGD needs a kick."""
    w0 = np.array([0.55, float(bump)])

    def g(w):
        return saddle_grad(w)

    path = run_first_order(w0, g, "sgd", float(lr), int(steps))
    print(
        f"start {w0}  end {np.round(path[-1], 3)}  "
        f"J {saddle_J(w0):.3f} → {saddle_J(path[-1]):.3f}"
    )
    print("y=0 — гребень. Если bump=0, SGD стоит на седле навсегда.")
    fig, ax = plt.subplots(figsize=(5.6, 5.2), layout="constrained")
    _contour(ax, saddle_J, (-2.2, 2.2), (-2.2, 2.2), levels=16)
    _draw_path(ax, path, COLORS["sgd"], "SGD")
    ax.plot(0, 0, "*", color="#d62728", ms=12, label="saddle")
    ax.axhline(0, color="#aaa", lw=0.6)
    ax.axvline(0, color="#aaa", lw=0.6)
    ax.set_title("saddle  J = (x² − y²)/2")
    ax.legend(fontsize=8)
    plt.show()


def cliff_clip(lr: float = 0.15, clip: float = 1.5, steps: float = 18.0) -> None:
    """Exponential wall. Unclipped SGD jumps off the map; clip stays on the path."""
    w0 = np.array([0.4, 0.85])
    clip_v = None if clip <= 0 else float(clip)
    path_raw = run_first_order(w0, cliff_grad, "sgd", float(lr), int(steps), clip=None)
    path_c = run_first_order(w0, cliff_grad, "sgd", float(lr), int(steps), clip=clip_v)
    g0 = cliff_grad(w0)
    print(
        f"‖g‖ at start = {np.linalg.norm(g0):.1f}   "
        f"unclipped end {np.round(path_raw[-1], 2)}   "
        f"clip={clip_v} end {np.round(path_c[-1], 2)}"
    )
    fig, ax = plt.subplots(figsize=(5.8, 5.0), layout="constrained")
    _contour(ax, cliff_J, (-2.0, 2.4), (-1.2, 1.8), levels=12)
    _draw_path(ax, path_raw, "#aaaaaa", "no clip")
    _draw_path(ax, path_c, COLORS["sgd"], f"clip={clip_v}")
    ax.set_title("cliff / exploding grad")
    ax.legend(fontsize=8)
    plt.show()


def adaptive_bowl(lr: float = 0.12, steps: float = 80.0) -> None:
    """AdaGrad / RMSProp / Adam on a κ=40 bowl. Same ε for all — that's the point."""
    hess = rotate_hessian(0.06, 2.4, deg=35.0)
    w0 = np.array([-1.1, 1.7])

    def g(w):
        return quadratic_grad(w, hess, W_STAR)

    fig, ax = plt.subplots(figsize=(5.8, 5.2), layout="constrained")
    _contour(ax, lambda w: quadratic_J(w, hess, W_STAR), (-2.0, 2.6), (-1.6, 2.4))
    ax.plot(*W_STAR, "*", color="#d62728", ms=11)
    print(f"{'method':10}  J_end")
    for name in ("sgd", "adagrad", "rmsprop", "adam"):
        path = run_first_order(w0, g, name, float(lr), int(steps))
        _draw_path(ax, path, COLORS[name], name)
        print(f"{name:10}  {quadratic_J(path[-1], hess, W_STAR):.4f}")
    ax.set_title("adaptive ε, same starting ε")
    ax.legend(loc="upper right", fontsize=8)
    plt.show()


def minibatch_diabetes(
    batch_size: int = 16, lr: float = 0.08, epochs: int = 40
) -> None:
    """Same linear model, different batch sizes. Full batch is smooth, tiny batch jitters."""
    from data import load_diabetes_split

    d = load_diabetes_split()
    X, y = d.X_train, d.y_train
    y = (y - y.mean()) / y.std()
    fig, ax = plt.subplots(figsize=(6.4, 3.6), layout="constrained")
    for bs, color in ((len(X), "#1f77b4"), (int(batch_size), "#d62728"), (1, "#aaaaaa")):
        _, losses = sgd_mse(X, y, float(lr), bs, int(epochs), seed=0)
        label = {len(X): "full batch", 1: "batch=1"}.get(bs, f"batch={bs}")
        ax.plot(losses, color=color, lw=1.6, label=label)
        print(f"{label:12}  last J={losses[-1]:.3f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train MSE")
    ax.set_title("diabetes linear SGD — batch size")
    ax.legend(fontsize=8)
    plt.show()


def newton_vs_sgd_diabetes(damp: float = 0.0, sgd_lr: float = 0.05) -> None:
    """One Newton step = OLS. SGD needs many epochs to get close."""
    from data import load_diabetes_split

    d = load_diabetes_split()
    X, y = d.X_train, d.y_train
    y = (y - y.mean()) / y.std()
    w_ols = newton_mse(X, y, damp=0.0, steps=1)[-1]
    w_damp = newton_mse(X, y, damp=float(damp), steps=1)[-1]
    w_sgd, losses = sgd_mse(X, y, float(sgd_lr), batch_size=32, epochs=80, seed=0)

    def mse(w):
        r = X @ w - y
        return 0.5 * float(r @ r) / len(X)

    print(f"OLS   (Newton, λ=0)   J={mse(w_ols):.4f}  ‖w‖={np.linalg.norm(w_ols):.2f}")
    print(f"damp  λ={damp:g}         J={mse(w_damp):.4f}  ‖w‖={np.linalg.norm(w_damp):.2f}")
    print(f"SGD   80 epochs        J={mse(w_sgd):.4f}  ‖w‖={np.linalg.norm(w_sgd):.2f}")
    fig, ax = plt.subplots(figsize=(6.4, 3.4), layout="constrained")
    ax.plot(losses, color=COLORS["sgd"], label="SGD")
    ax.axhline(mse(w_ols), color=COLORS["newton"], ls="--", label="Newton / OLS")
    if damp > 0:
        ax.axhline(mse(w_damp), color="#ff7f0e", ls=":", label=f"Newton λ={damp:g}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train MSE")
    ax.legend(fontsize=8)
    ax.set_title("diabetes — Newton vs SGD")
    plt.show()


def scoreboard(board: dict) -> None:
    if not board:
        print("board empty — run a Fashion cell first")
        return
    names = list(board)
    fig, ax = plt.subplots(figsize=(7.2, 3.4), layout="constrained")
    x = np.arange(len(names))
    ax.bar(x - 0.18, [board[n]["train"] for n in names], 0.36, label="train", color="#1f77b4")
    ax.bar(x + 0.18, [board[n]["test"] for n in names], 0.36, label="test", color="#d62728")
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.legend()
    ax.set_title("Fashion-MNIST")
    plt.show()
    print(f"{'name':16} train   test    gap")
    for n in names:
        r = board[n]
        print(f"{n:16} {r['train']:.3f}  {r['test']:.3f}  {r['gap']:.3f}")


def grad_bars(rows: list[tuple[str, list[tuple[str, float]]]]) -> None:
    """rows = [(label, [(layer, mean_abs_grad), ...]), ...]"""
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.0, 3.4), layout="constrained")
    width = 0.8 / len(rows)
    n_layers = len(rows[0][1])
    xs = np.arange(n_layers)
    for i, (label, pairs) in enumerate(rows):
        ax.bar(xs + i * width, [p[1] for p in pairs], width, label=label)
    ax.set_xticks(xs + width * (len(rows) - 1) / 2, [p[0] for p in rows[0][1]], rotation=20, ha="right")
    ax.set_ylabel("mean |grad|")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("gradient mass by layer")
    plt.show()
    for label, pairs in rows:
        print(label, " ".join(f"{n.split('.')[0]}={v:.2e}" for n, v in pairs))
