"""sklearn built-ins only — nothing is synthesized, nothing is downloaded.

- diabetes: linear L1/L2
- breast_cancer: logistic L1/L2
- digits 3-vs-8: MLP regularizers / bagging / sparse hidden
- digits 10-way 8x8: CNN, aug, FGSM
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits
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
    feature_names: tuple[str, ...] = ()


def _split_xy(
    X: np.ndarray,
    y: np.ndarray,
    n_train: int,
    n_val: int,
    seed: int,
    stratify: bool,
    scale: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    strat = y if stratify else None
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, train_size=n_train, stratify=strat, random_state=seed
    )
    strat_tmp = y_tmp if stratify else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, train_size=n_val, stratify=strat_tmp, random_state=seed + 1
    )
    if scale:
        scaler = StandardScaler().fit(X_train)
        X_train, X_val, X_test = (
            scaler.transform(X_train),
            scaler.transform(X_val),
            scaler.transform(X_test),
        )
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_diabetes_split(
    n_train: int = 80, n_val: int = 80, seed: int = 0
) -> Split:
    bunch = load_diabetes()
    Xtr, ytr, Xva, yva, Xte, yte = _split_xy(
        bunch.data.astype(np.float64),
        bunch.target.astype(np.float64),
        n_train,
        n_val,
        seed,
        stratify=False,
        scale=True,
    )
    return Split(
        Xtr, ytr, Xva, yva, Xte, yte, Xtr.shape[1], 1, tuple(bunch.feature_names)
    )


def load_cancer_split(
    n_train: int = 40, n_val: int = 80, seed: int = 0
) -> Split:
    bunch = load_breast_cancer()
    Xtr, ytr, Xva, yva, Xte, yte = _split_xy(
        bunch.data.astype(np.float32),
        bunch.target.astype(np.int64),
        n_train,
        n_val,
        seed,
        stratify=True,
        scale=True,
    )
    return Split(
        Xtr, ytr, Xva, yva, Xte, yte, Xtr.shape[1], 2, tuple(bunch.feature_names)
    )


def load_digits_pair(
    a: int = 3,
    b: int = 8,
    n_train: int = 50,
    n_val: int = 80,
    seed: int = 0,
) -> Split:
    """Real 8x8 digits, two classes. Small train split so a wide net can memorize."""
    digits = load_digits()
    mask = (digits.target == a) | (digits.target == b)
    X = (digits.data[mask] / 16.0).astype(np.float32)
    y = (digits.target[mask] == b).astype(np.int64)
    n_val = min(n_val, len(y) - n_train - 40)
    Xtr, ytr, Xva, yva, Xte, yte = _split_xy(
        X, y, n_train, n_val, seed, stratify=True, scale=False
    )
    return Split(Xtr, ytr, Xva, yva, Xte, yte, 64, 2, ())


def load_digits_split(
    n_train: int = 250,
    n_val: int = 250,
    seed: int = 0,
    as_images: bool = False,
) -> Split:
    digits = load_digits()
    X = (digits.data / 16.0).astype(np.float32)
    y = digits.target.astype(np.int64)
    Xtr, ytr, Xva, yva, Xte, yte = _split_xy(
        X, y, n_train, n_val, seed, stratify=True, scale=False
    )
    if as_images:
        Xtr = Xtr.reshape(-1, 1, 8, 8)
        Xva = Xva.reshape(-1, 1, 8, 8)
        Xte = Xte.reshape(-1, 1, 8, 8)
    return Split(Xtr, ytr, Xva, yva, Xte, yte, 64, 10, ())


def shift_images(images: np.ndarray, max_shift: int = 1, seed: int = 0) -> np.ndarray:
    """ch. 7.4: small translations. Do not flip — 6/9 would swap class."""
    rng = np.random.default_rng(seed)
    out = np.empty_like(images)
    for i, img in enumerate(images):
        dy = int(rng.integers(-max_shift, max_shift + 1))
        dx = int(rng.integers(-max_shift, max_shift + 1))
        out[i] = np.roll(np.roll(img, dy, axis=-2), dx, axis=-1)
    return out
