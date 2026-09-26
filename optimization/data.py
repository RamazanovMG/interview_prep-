"""Public datasets only. Nothing is synthesized.

- diabetes: sklearn built-in (linear MSE)
- Fashion-MNIST: official Zalando dump, reused from regularization/data_cache if present
"""

from __future__ import annotations

import gzip
import struct
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


def fashion_cache() -> Path:
    here = Path(__file__).resolve().parent / "data_cache" / "fashion-mnist"
    sibling = (
        Path(__file__).resolve().parent.parent
        / "regularization"
        / "data_cache"
        / "fashion-mnist"
    )
    marker = "train-images-idx3-ubyte.gz"
    for cand in (sibling, here):
        if (cand / marker).exists() and (cand / marker).stat().st_size > 0:
            return cand
    here.mkdir(parents=True, exist_ok=True)
    return here


def ensure_fashion_mnist() -> Path:
    cache = fashion_cache()
    cache.mkdir(parents=True, exist_ok=True)
    for name in FASHION_FILES:
        dest = cache / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        url = FASHION_URL + name
        print(f"downloading {url}", file=sys.stderr)
        urllib.request.urlretrieve(url, dest)
    return cache


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


def load_diabetes_split(
    n_train: int = 200, n_val: int = 80, seed: int = 0
) -> Split:
    bunch = load_diabetes()
    X = bunch.data.astype(np.float64)
    y = bunch.target.astype(np.float64)
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
    return Split(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_train.shape[1],
        1,
        tuple(bunch.feature_names),
    )


def load_fashion_split(
    n_train: int = 800,
    n_val: int = 600,
    n_test: int = 1200,
    seed: int = 0,
) -> Split:
    cache = ensure_fashion_mnist()
    X_all = _read_idx(cache / "train-images-idx3-ubyte.gz")
    y_all = _read_idx(cache / "train-labels-idx1-ubyte.gz")
    X_te_all = _read_idx(cache / "t10k-images-idx3-ubyte.gz")
    y_te_all = _read_idx(cache / "t10k-labels-idx1-ubyte.gz")

    X_tv, y_tv = _stratified_take(X_all, y_all, n_train + n_val, seed)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tv, y_tv, train_size=n_train, stratify=y_tv, random_state=seed
    )
    X_te, y_te = _stratified_take(X_te_all, y_te_all, n_test, seed + 3)

    def _prep(X: np.ndarray) -> np.ndarray:
        return (X.astype(np.float32) / 255.0).reshape(len(X), -1)

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
