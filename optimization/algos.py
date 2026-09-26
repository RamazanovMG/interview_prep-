"""First-order and Newton steps from Goodfellow et al., ch. 8.

All updates are NumPy, so the 2-D geometry cells stay instant.
Notation matches the book: ε = learning rate, β = momentum.
"""

from __future__ import annotations

import numpy as np


def sgd(w: np.ndarray, g: np.ndarray, lr: float) -> np.ndarray:
    """θ ← θ − ε g.  Eq. 8.3 / 8.13."""
    return w - lr * g


def momentum_update(
    w: np.ndarray, vel: np.ndarray, g: np.ndarray, lr: float, beta: float
) -> tuple[np.ndarray, np.ndarray]:
    """v ← β v − ε g;  θ ← θ + v.  Eq. 8.15–8.16."""
    vel = beta * vel - lr * g
    return w + vel, vel


def adagrad(
    w: np.ndarray, g: np.ndarray, acc: np.ndarray, lr: float, eps: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate g², scale each coordinate.  §8.5.1."""
    acc = acc + g * g
    return w - lr * g / (np.sqrt(acc) + eps), acc


def rmsprop(
    w: np.ndarray,
    g: np.ndarray,
    s: np.ndarray,
    lr: float,
    rho: float = 0.9,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Leaky AdaGrad.  §8.5.2."""
    s = rho * s + (1.0 - rho) * g * g
    return w - lr * g / (np.sqrt(s) + eps), s


def adam(
    w: np.ndarray,
    g: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    t: int,
    lr: float,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RMSProp + momentum + bias correction.  §8.5.3."""
    m = b1 * m + (1.0 - b1) * g
    v = b2 * v + (1.0 - b2) * g * g
    mhat = m / (1.0 - b1**t)
    vhat = v / (1.0 - b2**t)
    return w - lr * mhat / (np.sqrt(vhat) + eps), m, v


def newton(
    w: np.ndarray, g: np.ndarray, hess: np.ndarray, damp: float = 0.0
) -> np.ndarray:
    """θ ← θ − (H + λI)⁻¹ g.  Damping = Levenberg when H is not PD.  §8.6.1."""
    n = hess.shape[0]
    return w - np.linalg.solve(hess + damp * np.eye(n), g)


def clip_grad(g: np.ndarray, max_norm: float) -> np.ndarray:
    """Rescale if ‖g‖ > max_norm. Direction stays, length is capped.  §8.2.4."""
    nrm = float(np.linalg.norm(g))
    if nrm > max_norm and nrm > 0:
        return g * (max_norm / nrm)
    return g


def rotate_hessian(lmin: float, lmax: float, deg: float = 0.0) -> np.ndarray:
    """2×2 Hessian with eigenvalues (lmax, lmin), then rotated."""
    th = np.deg2rad(deg)
    q = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return q @ np.diag([lmax, lmin]) @ q.T


def quadratic_J(w: np.ndarray, hess: np.ndarray, w_star: np.ndarray) -> float:
    d = w - w_star
    return 0.5 * float(d @ hess @ d)


def quadratic_grad(w: np.ndarray, hess: np.ndarray, w_star: np.ndarray) -> np.ndarray:
    return hess @ (w - w_star)


def saddle_J(w: np.ndarray) -> float:
    return 0.5 * float(w[0] ** 2 - w[1] ** 2)


def saddle_grad(w: np.ndarray) -> np.ndarray:
    return np.array([w[0], -w[1]], dtype=float)


def cliff_J(w: np.ndarray) -> float:
    """Gentle bowl + an exponential wall. Classic exploding-grad toy."""
    x, y = float(w[0]), float(w[1])
    return 0.5 * x * x + 0.05 * y * y + 0.12 * np.exp(8.0 * (y - 0.45))


def cliff_grad(w: np.ndarray) -> np.ndarray:
    x, y = float(w[0]), float(w[1])
    return np.array([x, 0.1 * y + 0.12 * 8.0 * np.exp(8.0 * (y - 0.45))])


def safe_lr(lmax: float) -> float:
    """Largest ε that still converges on a quadratic: ε < 2 / λ_max."""
    return 2.0 / max(lmax, 1e-12)


def run_first_order(
    w0: np.ndarray,
    grad_fn,
    method: str,
    lr: float,
    steps: int,
    beta: float = 0.9,
    rho: float = 0.9,
    clip: float | None = None,
) -> np.ndarray:
    """Return (steps+1, d) trajectory including the start point."""
    w = np.asarray(w0, dtype=float).copy()
    vel = np.zeros_like(w)
    acc = np.zeros_like(w)
    m = np.zeros_like(w)
    s = np.zeros_like(w)
    hist = [w.copy()]
    for t in range(1, steps + 1):
        g = grad_fn(w + beta * vel) if method == "nesterov" else grad_fn(w)
        if clip is not None:
            g = clip_grad(g, float(clip))
        if method == "sgd":
            w = sgd(w, g, lr)
        elif method in ("momentum", "nesterov"):
            w, vel = momentum_update(w, vel, g, lr, beta)
        elif method == "adagrad":
            w, acc = adagrad(w, g, acc, lr)
        elif method == "rmsprop":
            w, s = rmsprop(w, g, s, lr, rho)
        elif method == "adam":
            w, m, s = adam(w, g, m, s, t, lr)
        else:
            raise ValueError(method)
        hist.append(w.copy())
    return np.stack(hist)


def sgd_mse(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    batch_size: int,
    epochs: int,
    seed: int = 0,
    method: str = "sgd",
    beta: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear MSE on a real design matrix. Returns (w, per-epoch full-batch loss)."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    vel = np.zeros(d)
    losses = np.empty(epochs, dtype=float)
    bs = max(1, min(int(batch_size), n))
    for ep in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            xb = X[perm[i : i + bs]]
            yb = y[perm[i : i + bs]]
            r = xb @ w - yb
            g = (xb.T @ r) / len(xb)
            if method == "sgd":
                w = sgd(w, g, lr)
            else:
                w, vel = momentum_update(w, vel, g, lr, beta)
        r = X @ w - y
        losses[ep] = 0.5 * float(r @ r) / n
    return w, losses


def newton_mse(
    X: np.ndarray, y: np.ndarray, damp: float = 0.0, steps: int = 1
) -> list[np.ndarray]:
    """Newton on linear least squares. One undamped step is the exact OLS solution."""
    n, d = X.shape
    xtx = X.T @ X / n
    xty = X.T @ y / n
    w = np.zeros(d, dtype=float)
    hist = [w.copy()]
    for _ in range(steps):
        g = xtx @ w - xty
        w = newton(w, g, xtx, damp=damp)
        hist.append(w.copy())
    return hist
