"""Closed-form L1/L2 results from Goodfellow et al., ch. 7.1."""

from __future__ import annotations

import numpy as np


def ridge_normal_equation(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Eq. 7.17: w = (X^T X + α I)^{-1} X^T y. No bias term."""
    n_features = X.shape[1]
    xtx = X.T @ X
    xty = X.T @ y
    return np.linalg.solve(xtx + alpha * np.eye(n_features), xty)


def l2_shrink_eigen(w_star: np.ndarray, hessian: np.ndarray, alpha: float) -> np.ndarray:
    """Eq. 7.13: ŵ = Q (Λ + αI)^{-1} Λ Q^T w*.

    Directions with λ_i ≪ α are killed; λ_i ≫ α are left almost alone.
    """
    eigvals, eigvecs = np.linalg.eigh(hessian)
    scale = eigvals / (eigvals + alpha)
    return eigvecs @ (scale * (eigvecs.T @ w_star))


def l1_soft_threshold(w_star: np.ndarray, hessian_diag: np.ndarray, alpha: float) -> np.ndarray:
    """Eq. 7.23: ŵ_i = sign(w*_i) max(|w*_i| - α / H_ii, 0).

    Assumes a diagonal positive-definite Hessian (uncorrelated features / PCA).
    L2 never hits exact zero if w*_i ≠ 0; L1 does once α ≥ |w*_i| H_ii.
    """
    threshold = alpha / hessian_diag
    return np.sign(w_star) * np.maximum(np.abs(w_star) - threshold, 0.0)


def l2_shrink_diag(w_star: np.ndarray, hessian_diag: np.ndarray, alpha: float) -> np.ndarray:
    """Diagonal special case of eq. 7.13: ŵ_i = H_ii / (H_ii + α) w*_i."""
    return (hessian_diag / (hessian_diag + alpha)) * w_star
