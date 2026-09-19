"""Public datasets only. Nothing is synthesized.

- diabetes / breast_cancer: sklearn built-ins (linear)
- Fashion-MNIST: auto-downloaded from Zalando's GitHub on first run
  https://github.com/zalandoresearch/fashion-mnist
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CACHE = Path(__file__).resolve().parent / "data_cache" / "fashion-mnist"
FASHION_URL = (
    "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
)
FASHION_FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)
FASHION_LABELS = (
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


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


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        magic = struct.unpack(">I", fh.read(4))[0]
        if magic == 2049:
            struct.unpack(">I", fh.read(4))
            return np.frombuffer(fh.read(), dtype=np.uint8).copy()
        if magic == 2051:
            n, rows, cols = struct.unpack(">III", fh.read(12))
            return np.frombuffer(fh.read(), dtype=np.uint8).reshape(n, rows, cols).copy()
        raise ValueError(f"bad idx magic {magic} in {path}")


def ensure_fashion_mnist() -> Path:
    """Fetch the 4 official gz files if missing (~30MB)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in FASHION_FILES:
        dest = CACHE / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        url = FASHION_URL + name
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return CACHE


def _stratified_take(
    X: np.ndarray, y: np.ndarray, n: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    per = n // int(y.max() + 1)
    picks = []
    for c in range(int(y.max()) + 1):
        idx = np.flatnonzero(y == c)
        picks.append(rng.choice(idx, size=min(per, len(idx)), replace=False))
    picks = np.concatenate(picks)
    rng.shuffle(picks)
    return X[picks], y[picks]


def load_fashion_split(
    n_train: int = 1500,
    n_val: int = 1000,
    n_test: int = 2000,
    seed: int = 0,
    as_images: bool = False,
) -> Split:
    """Official Fashion-MNIST, small stratified subset so a wide net can overfit."""
    ensure_fashion_mnist()
    X_all = _read_idx(CACHE / "train-images-idx3-ubyte.gz")
    y_all = _read_idx(CACHE / "train-labels-idx1-ubyte.gz")
    X_te_all = _read_idx(CACHE / "t10k-images-idx3-ubyte.gz")
    y_te_all = _read_idx(CACHE / "t10k-labels-idx1-ubyte.gz")

    X_tv, y_tv = _stratified_take(X_all, y_all, n_train + n_val, seed)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tv, y_tv, train_size=n_train, stratify=y_tv, random_state=seed
    )
    X_te, y_te = _stratified_take(X_te_all, y_te_all, n_test, seed + 3)

    def _prep(X: np.ndarray) -> np.ndarray:
        x = (X.astype(np.float32) / 255.0)
        if as_images:
            return x[:, None, :, :]
        return x.reshape(len(x), -1)

    return Split(
        _prep(X_tr),
        y_tr.astype(np.int64),
        _prep(X_va),
        y_va.astype(np.int64),
        _prep(X_te),
        y_te.astype(np.int64),
        784,
        10,
        FASHION_LABELS,
    )


def shift_images(images: np.ndarray, max_shift: int = 1, seed: int = 0) -> np.ndarray:
    """ch. 7.4: small translations. Do not flip — 6/9 would swap class."""
    rng = np.random.default_rng(seed)
    out = np.empty_like(images)
    for i, img in enumerate(images):
        dy = int(rng.integers(-max_shift, max_shift + 1))
        dx = int(rng.integers(-max_shift, max_shift + 1))
        out[i] = np.roll(np.roll(img, dy, axis=-2), dx, axis=-1)
    return out
