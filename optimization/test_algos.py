"""Sanity checks for ch. 8 updates. Run: python test_algos.py"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from algos import (
    cliff_grad,
    clip_grad,
    newton,
    newton_mse,
    quadratic_J,
    quadratic_grad,
    rotate_hessian,
    run_first_order,
    safe_lr,
    sgd,
    sgd_mse,
)


def main() -> None:
    w_star = np.array([1.0, -0.5])
    hess = np.diag([2.0, 0.4])

    def g(w):
        return quadratic_grad(w, hess, w_star)

    # Newton on a quadratic: one step from anywhere.
    w0 = np.array([-3.0, 4.0])
    w1 = newton(w0, g(w0), hess, damp=0.0)
    assert np.allclose(w1, w_star, atol=1e-8), w1

    # Safe lr bound: ε = 2/λmax oscillates, slightly below converges.
    cap = safe_lr(2.0)
    assert np.isclose(cap, 1.0)
    path_ok = run_first_order(w0, g, "sgd", lr=0.6, steps=80)
    assert quadratic_J(path_ok[-1], hess, w_star) < 1e-4

    path_bad = run_first_order(w0, g, "sgd", lr=1.2, steps=20)
    assert quadratic_J(path_bad[-1], hess, w_star) > quadratic_J(w0, hess, w_star)

    # Momentum should beat SGD on a skinny bowl in a fixed step budget.
    skinny = rotate_hessian(0.05, 2.0, deg=0.0)
    w0 = np.array([1.5, 1.5])

    def gs(w):
        return quadratic_grad(w, skinny, np.zeros(2))

    sgd_p = run_first_order(w0, gs, "sgd", lr=0.15, steps=40)
    mom_p = run_first_order(w0, gs, "momentum", lr=0.15, steps=40, beta=0.9)
    assert quadratic_J(mom_p[-1], skinny, np.zeros(2)) < quadratic_J(
        sgd_p[-1], skinny, np.zeros(2)
    )

    # Clip only shrinks, never flips.
    gbig = np.array([3.0, 4.0])
    c = clip_grad(gbig, 2.5)
    assert np.isclose(np.linalg.norm(c), 2.5)
    assert np.allclose(c / np.linalg.norm(c), gbig / np.linalg.norm(gbig))

    w_cliff = np.array([0.4, 0.85])
    raw = run_first_order(w_cliff, cliff_grad, "sgd", lr=0.15, steps=6, clip=None)
    held = run_first_order(w_cliff, cliff_grad, "sgd", lr=0.15, steps=6, clip=1.5)
    assert np.abs(raw[:, 1]).max() > 2.0
    assert np.abs(held[:, 1]).max() < 1.2

    # Linear MSE: undamped Newton = OLS.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 5))
    w_true = rng.normal(size=5)
    y = X @ w_true
    w_n = newton_mse(X, y, damp=0.0, steps=1)[-1]
    w_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.allclose(w_n, w_ols, atol=1e-8)

    w_s, losses = sgd_mse(X, y, lr=0.05, batch_size=32, epochs=40, seed=0)
    assert losses[-1] < losses[0]
    assert np.linalg.norm(w_s - w_ols) < 0.4

    # Plain SGD step.
    assert np.allclose(sgd(np.array([1.0, 2.0]), np.array([1.0, 0.0]), 0.5), [0.5, 2.0])
    print("algo checks ok")


if __name__ == "__main__":
    main()
