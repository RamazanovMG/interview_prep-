"""Sanity checks for the ch. 7.1 closed forms. Run: python test_closed_form.py"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from closed_form import l1_soft_threshold, l2_shrink_diag, l2_shrink_eigen, ridge_normal_equation


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6))
    w_true = rng.normal(size=6)
    y = X @ w_true

    w0 = ridge_normal_equation(X, y, alpha=0.0)
    w_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.allclose(w0, w_ols, atol=1e-6), (w0, w_ols)

    w_ridge = ridge_normal_equation(X, y, alpha=3.0)
    assert np.linalg.norm(w_ridge) < np.linalg.norm(w0)

    H = np.diag([0.1, 2.0])
    w_star = np.array([2.0, 1.0])
    shrunk = l2_shrink_eigen(w_star, H, alpha=0.5)
    # poorly determined axis (λ=0.1) shrinks more
    rel = np.abs(shrunk / w_star)
    assert rel[0] < rel[1]
    assert np.allclose(shrunk, l2_shrink_diag(w_star, np.diag(H), 0.5))

    soft = l1_soft_threshold(np.array([0.2, 1.5]), np.array([1.0, 1.0]), alpha=0.5)
    assert soft[0] == 0.0
    assert np.isclose(soft[1], 1.0)
    print("closed-form checks ok")


if __name__ == "__main__":
    main()
