"""Synthetic + sklearn datasets sized so an unregularized net can overfit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits, make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    n_features: int
    n_classes: int


def _as_float32(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple(a.astype(np.float32) for a in arrays)


def make_sparse_regression(
    n_train: int = 80,
    n_val: int = 80,
    n_test: int = 400,
    n_features: int = 120,
    n_informative: int = 12,
    noise: float = 0.5,
    seed: int = 0,
) -> tuple[Split, np.ndarray]:
    """High-p, low-n regression. Only `n_informative` weights are nonzero."""
    rng = np.random.default_rng(seed)
    n = n_train + n_val + n_test
    X = rng.normal(size=(n, n_features))
    true_w = np.zeros(n_features, dtype=np.float64)
    true_w[:n_informative] = rng.normal(scale=1.5, size=n_informative)
    y = X @ true_w + rng.normal(scale=noise, size=n)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, train_size=n_train, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, train_size=n_val, random_state=seed + 1
    )
    scaler = StandardScaler().fit(X_train)
    X_train, X_val, X_test = (
        scaler.transform(X_train),
        scaler.transform(X_val),
        scaler.transform(X_test),
    )
    X_train, X_val, X_test = _as_float32(X_train, X_val, X_test)
    y_train, y_val, y_test = _as_float32(y_train, y_val, y_test)
    split = Split(X_train, y_train, X_val, y_val, X_test, y_test, n_features, 1)
    return split, true_w


def make_sparse_classification(
    n_train: int = 120,
    n_val: int = 80,
    n_test: int = 400,
    n_features: int = 80,
    n_informative: int = 8,
    seed: int = 0,
) -> Split:
    X, y = make_classification(
        n_samples=n_train + n_val + n_test,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        class_sep=1.2,
        flip_y=0.05,
        random_state=seed,
    )
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, train_size=n_train, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, train_size=n_val, stratify=y_tmp, random_state=seed + 1
    )
    scaler = StandardScaler().fit(X_train)
    X_train, X_val, X_test = _as_float32(
        scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)
    )
    return Split(
        X_train,
        y_train.astype(np.int64),
        X_val,
        y_val.astype(np.int64),
        X_test,
        y_test.astype(np.int64),
        n_features,
        2,
    )


def make_overfit_moons(
    n_train: int = 70,
    n_val: int = 80,
    n_test: int = 400,
    noise: float = 0.28,
    label_flip: float = 0.12,
    seed: int = 0,
) -> Split:
    """2-D moons, tiny train set + label noise → a wide MLP memorizes."""
    rng = np.random.default_rng(seed)
    X, y = make_moons(
        n_samples=n_train + n_val + n_test, noise=noise, random_state=seed
    )
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, train_size=n_train, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, train_size=n_val, stratify=y_tmp, random_state=seed + 1
    )
    flip = rng.random(len(y_train)) < label_flip
    y_train = y_train.copy()
    y_train[flip] = 1 - y_train[flip]
    return Split(
        X_train.astype(np.float32),
        y_train.astype(np.int64),
        X_val.astype(np.float32),
        y_val.astype(np.int64),
        X_test.astype(np.float32),
        y_test.astype(np.int64),
        2,
        2,
    )


def make_digits_split(
    n_train: int = 250,
    n_val: int = 250,
    seed: int = 0,
    as_images: bool = False,
) -> Split:
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, train_size=n_train, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, train_size=n_val, stratify=y_tmp, random_state=seed + 1
    )
    if as_images:
        X_train = X_train.reshape(-1, 1, 8, 8)
        X_val = X_val.reshape(-1, 1, 8, 8)
        X_test = X_test.reshape(-1, 1, 8, 8)
    return Split(X_train, y_train, X_val, y_val, X_test, y_test, 64, 10)


def shift_images(images: np.ndarray, max_shift: int = 1, seed: int = 0) -> np.ndarray:
    """§7.4: small translations. Do not flip — 6/9 would swap class."""
    rng = np.random.default_rng(seed)
    out = np.empty_like(images)
    for i, img in enumerate(images):
        dy = int(rng.integers(-max_shift, max_shift + 1))
        dx = int(rng.integers(-max_shift, max_shift + 1))
        out[i] = np.roll(np.roll(img, dy, axis=-2), dx, axis=-1)
    return out
