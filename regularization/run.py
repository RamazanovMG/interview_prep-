#!/usr/bin/env python3
"""Run Goodfellow ch. 7 regularization experiments and write results/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from closed_form import ridge_normal_equation  # noqa: E402
from data import (  # noqa: E402
    FASHION_LABELS,
    load_cancer_split,
    load_diabetes_split,
    load_fashion_split,
    shift_images,
)
from models import MLP, TinyCNN, fgsm, n_params  # noqa: E402
from plots import (  # noqa: E402
    RESULTS,
    plot_adversarial,
    plot_comparison_bars,
    plot_constraint_sets,
    plot_decision_boundaries,
    plot_l2_geometry,
    plot_learning_curves,
    plot_linear_mse,
    plot_shrinkage_1d,
    plot_fashion_grid,
    plot_sparsity_hist,
    plot_weight_stems,
)
from train import (  # noqa: E402
    TrainConfig,
    bagged_proba,
    bootstrap_indices,
    metrics_for,
    train_classifier,
)
import torch  # noqa: E402


def _jsonable(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        clean = {}
        for k, v in row.items():
            if isinstance(v, (np.floating, float)):
                clean[k] = round(float(v), 4)
            elif isinstance(v, (np.integer, int)):
                clean[k] = int(v)
            else:
                clean[k] = v
        out.append(clean)
    return out


def exp_geometry() -> list[Path]:
    return [plot_l2_geometry(), plot_constraint_sets(), plot_shrinkage_1d()]


def exp_linear() -> list[dict]:
    split = load_diabetes_split()
    ols = LinearRegression().fit(split.X_train, split.y_train)
    ridge = Ridge(alpha=2.0).fit(split.X_train, split.y_train)
    lasso = Lasso(alpha=0.8, max_iter=20000).fit(split.X_train, split.y_train)
    closed = ridge_normal_equation(
        np.c_[split.X_train, np.ones(len(split.X_train))],
        split.y_train,
        alpha=0.0,
    )
    # closed-form unregularized (with bias column) should track OLS
    align = float(
        np.dot(closed[:-1], ols.coef_)
        / (np.linalg.norm(closed[:-1]) * np.linalg.norm(ols.coef_) + 1e-12)
    )

    estimates = {
        "OLS  (α=0)": ols.coef_,
        "Ridge L2  α=2": ridge.coef_,
        "Lasso L1  α=0.8": lasso.coef_,
    }
    plot_weight_stems(estimates, feature_names=split.feature_names)

    def mse(model, X, y) -> float:
        return float(np.mean((model.predict(X) - y) ** 2))

    rows = []
    for name, model in [
        ("OLS", ols),
        ("Ridge L2", ridge),
        ("Lasso L1", lasso),
    ]:
        w = model.coef_
        rows.append(
            {
                "experiment": "diabetes_regression",
                "name": name,
                "test_mse": mse(model, split.X_test, split.y_test),
                "n_zeros": int(np.sum(np.abs(w) < 1e-6)),
                "ols_closed_form_align": align,
            }
        )
    plot_linear_mse(rows)

    clf = load_cancer_split()
    logregs = {
        "logreg none": LogisticRegression(C=np.inf, max_iter=4000, random_state=0),
        "logreg L2": LogisticRegression(C=0.5, l1_ratio=0.0, max_iter=4000, random_state=0),
        "logreg L1": LogisticRegression(
            C=0.8, l1_ratio=1.0, solver="saga", max_iter=8000, random_state=0
        ),
    }
    cancer_weights = {}
    for name, model in logregs.items():
        model.fit(clf.X_train, clf.y_train)
        cancer_weights[name] = model.coef_.ravel()
        rows.append(
            {
                "experiment": "cancer_logreg",
                "name": name,
                "train_acc": float(model.score(clf.X_train, clf.y_train)),
                "test_acc": float(model.score(clf.X_test, clf.y_test)),
                "gap": float(
                    model.score(clf.X_train, clf.y_train)
                    - model.score(clf.X_test, clf.y_test)
                ),
                "n_zeros": int(np.sum(np.abs(model.coef_) < 1e-4)),
            }
        )
    plot_weight_stems(
        cancer_weights, path="cancer_weights.png", feature_names=clf.feature_names
    )
    return rows


def _mlp(
    in_dim: int,
    n_classes: int,
    dropout: float = 0.0,
    seed: int = 0,
    hidden: tuple[int, ...] = (256, 256),
    activation: str = "relu",
) -> MLP:
    torch.manual_seed(seed)
    return MLP(in_dim, n_classes, hidden=hidden, dropout=dropout, activation=activation)


def exp_mlp_digits(epochs: int) -> tuple[list[dict], dict]:
    from sklearn.decomposition import PCA

    data = load_fashion_split()
    plot_fashion_grid(data.X_train, data.y_train, FASHION_LABELS)
    pca = PCA(2, random_state=0).fit(data.X_train)
    base = dict(epochs=epochs, lr=0.003, batch_size=64, optimizer="adamw")
    specs = {
        "none": (0.0, TrainConfig(**base, restore_best=False, seed=0)),
        "l2": (0.0, TrainConfig(**base, weight_decay=0.04, seed=1)),
        "l1": (0.0, TrainConfig(**base, l1=8e-4, seed=2)),
        "dropout": (0.5, TrainConfig(**base, seed=3)),
        "input_noise": (0.0, TrainConfig(**base, input_noise=0.15, seed=4)),
        "early_stop": (
            0.0,
            TrainConfig(
                **base,
                early_stop_patience=20,
                restore_best=True,
                seed=5,
            ),
        ),
        "label_smooth": (0.0, TrainConfig(**base, label_smoothing=0.1, seed=6)),
        "l2+dropout": (0.4, TrainConfig(**base, weight_decay=0.02, seed=7)),
    }
    models = {}
    histories = {}
    rows = []
    for name, (dropout, cfg) in specs.items():
        model = _mlp(784, 10, dropout=dropout, seed=cfg.seed)
        hist = train_classifier(model, data.X_train, data.y_train, data.X_val, data.y_val, cfg)
        met = metrics_for(
            model, data.X_train, data.y_train, data.X_val, data.y_val, data.X_test, data.y_test
        )
        models[name] = model
        histories[name] = hist
        rows.append(
            {
                "experiment": "mlp_fashion",
                "name": name,
                "train_acc": met.train_acc,
                "val_acc": met.val_acc,
                "test_acc": met.test_acc,
                "gap": met.gap,
                "best_epoch": hist.best_epoch,
                "stopped_epoch": hist.stopped_epoch,
            }
        )
    plot_decision_boundaries(models, data.X_train, data.y_train, pca=pca)
    plot_learning_curves(
        {k: histories[k] for k in ("none", "l2", "dropout", "early_stop")}
    )
    plot_comparison_bars(
        rows,
        "comparison_bars.png",
        title="256-256 MLP on Fashion-MNIST (n_train=1500)",
    )
    return rows, histories


def exp_cnn_and_aug(epochs: int) -> list[dict]:
    flat = load_fashion_split(as_images=False, seed=1)
    images = load_fashion_split(as_images=True, seed=1)
    rows = []

    mlp = MLP(784, 10, hidden=(256, 256), dropout=0.0)
    hist = train_classifier(
        mlp,
        flat.X_train,
        flat.y_train,
        flat.X_val,
        flat.y_val,
        TrainConfig(epochs=epochs, lr=0.003, batch_size=64, optimizer="adamw", seed=0),
    )
    met = metrics_for(
        mlp, flat.X_train, flat.y_train, flat.X_val, flat.y_val, flat.X_test, flat.y_test
    )
    rows.append(
        {
            "experiment": "fashion_cnn",
            "name": "mlp",
            "params": n_params(mlp),
            "train_acc": met.train_acc,
            "test_acc": met.test_acc,
            "gap": met.gap,
            "stopped_epoch": hist.stopped_epoch,
        }
    )

    cnn = TinyCNN(img_size=28)
    train_classifier(
        cnn,
        images.X_train,
        images.y_train,
        images.X_val,
        images.y_val,
        TrainConfig(epochs=epochs, lr=0.003, batch_size=64, optimizer="adamw", seed=1),
    )
    met = metrics_for(
        cnn,
        images.X_train,
        images.y_train,
        images.X_val,
        images.y_val,
        images.X_test,
        images.y_test,
    )
    rows.append(
        {
            "experiment": "fashion_cnn",
            "name": "cnn  (param sharing)",
            "params": n_params(cnn),
            "train_acc": met.train_acc,
            "test_acc": met.test_acc,
            "gap": met.gap,
        }
    )

    X_aug = np.concatenate(
        [
            images.X_train,
            shift_images(images.X_train, max_shift=2, seed=11),
            shift_images(images.X_train, max_shift=2, seed=12),
        ]
    )
    y_aug = np.concatenate([images.y_train, images.y_train, images.y_train])
    cnn_aug = TinyCNN(img_size=28)
    train_classifier(
        cnn_aug,
        X_aug,
        y_aug,
        images.X_val,
        images.y_val,
        TrainConfig(epochs=epochs, lr=0.003, batch_size=64, optimizer="adamw", seed=2),
    )
    met = metrics_for(
        cnn_aug,
        images.X_train,
        images.y_train,
        images.X_val,
        images.y_val,
        images.X_test,
        images.y_test,
    )
    rows.append(
        {
            "experiment": "fashion_cnn",
            "name": "cnn + shift aug",
            "params": n_params(cnn_aug),
            "train_acc": met.train_acc,
            "test_acc": met.test_acc,
            "gap": met.gap,
        }
    )
    nice = {
        "mlp": "mlp",
        "cnn  (param sharing)": "cnn",
        "cnn + shift aug": "cnn+aug",
    }
    labeled = [
        {**r, "name": f"{nice.get(r['name'], r['name'])} {r['params']//1000}k"}
        for r in rows
    ]
    plot_comparison_bars(
        labeled,
        "digits_cnn_vs_mlp.png",
        title="Fashion-MNIST: sharing + 2px shifts",
    )
    return rows


def exp_bagging(epochs: int, n_models: int = 5) -> list[dict]:
    data = load_fashion_split(seed=3)
    single = _mlp(784, 10, seed=10)
    train_classifier(
        single,
        data.X_train,
        data.y_train,
        data.X_val,
        data.y_val,
            TrainConfig(epochs=epochs, lr=0.003, batch_size=64, optimizer="adamw", seed=10),
    )
    single_met = metrics_for(
        single, data.X_train, data.y_train, data.X_val, data.y_val, data.X_test, data.y_test
    )

    members = []
    for i in range(n_models):
        idx = bootstrap_indices(len(data.X_train), seed=20 + i)
        m = _mlp(784, 10, seed=20 + i)
        train_classifier(
            m,
            data.X_train[idx],
            data.y_train[idx],
            data.X_val,
            data.y_val,
            TrainConfig(epochs=epochs, lr=0.003, batch_size=64, optimizer="adamw", seed=20 + i),
        )
        members.append(m)

    bag_pred = bagged_proba(members, data.X_test).argmax(1)
    bag_train = bagged_proba(members, data.X_train).argmax(1)
    bag_test_acc = float(np.mean(bag_pred == data.y_test))
    bag_train_acc = float(np.mean(bag_train == data.y_train))
    rows = [
        {
            "experiment": "bagging",
            "name": "single mlp",
            "train_acc": single_met.train_acc,
            "test_acc": single_met.test_acc,
            "gap": single_met.gap,
        },
        {
            "experiment": "bagging",
            "name": f"bagging x{n_models}",
            "train_acc": bag_train_acc,
            "test_acc": bag_test_acc,
            "gap": bag_train_acc - bag_test_acc,
        },
    ]
    plot_comparison_bars(rows, "bagging.png", title="Ch. 7.11: bag of bootstrap MLPs")
    return rows


def exp_sparse_hidden(epochs: int) -> list[dict]:
    data = load_fashion_split(seed=4)
    rows = []
    hiddens = {}
    for name, act_l1 in [("no act. penalty", 0.0), ("L1 on hidden", 0.15)]:
        model = _mlp(784, 10, seed=30, activation="tanh", hidden=(64, 64))
        train_classifier(
            model,
            data.X_train,
            data.y_train,
            data.X_val,
            data.y_val,
            TrainConfig(
                epochs=epochs,
                lr=0.003,
                batch_size=64,
                optimizer="adamw",
                activation_l1=act_l1,
                seed=30,
            ),
        )
        model.eval()
        with torch.no_grad():
            _, h = model(torch.from_numpy(data.X_test), return_hidden=True)
        hiddens[name] = h.numpy()
        h_np = h.numpy()
        met = metrics_for(
            model, data.X_train, data.y_train, data.X_val, data.y_val, data.X_test, data.y_test
        )
        rows.append(
            {
                "experiment": "sparse_hidden",
                "name": name,
                "train_acc": met.train_acc,
                "test_acc": met.test_acc,
                "gap": met.gap,
                "frac_hidden_near_zero": float(np.mean(np.abs(h_np) < 1e-2)),
                "mean_abs_hidden": float(np.mean(np.abs(h_np))),
            }
        )
    plot_sparsity_hist(hiddens["no act. penalty"], hiddens["L1 on hidden"])
    return rows


@torch.no_grad()
def _acc(model, X, y) -> float:
    logits = model(torch.from_numpy(np.ascontiguousarray(X)))
    return float((logits.argmax(1).numpy() == y).mean())


def _adv_acc(model, X, y, eps: float, batch: int = 128) -> float:
    model.eval()
    correct = 0
    n = 0
    xt = torch.from_numpy(np.ascontiguousarray(X))
    yt = torch.from_numpy(np.ascontiguousarray(y)).long()
    for i in range(0, len(X), batch):
        xb, yb = xt[i : i + batch], yt[i : i + batch]
        adv = fgsm(model, xb, yb, eps)
        with torch.no_grad():
            pred = model(adv).argmax(1)
        correct += int((pred == yb).sum())
        n += yb.size(0)
    return correct / n


def exp_adversarial(epochs: int, eps: float = 0.18) -> list[dict]:
    data = load_fashion_split(as_images=True, seed=5)
    rows = []
    for name, adv_eps in [("clean train", 0.0), ("FGSM train", eps)]:
        model = TinyCNN(img_size=28)
        train_classifier(
            model,
            data.X_train,
            data.y_train,
            data.X_val,
            data.y_val,
            TrainConfig(
                epochs=epochs,
                lr=0.003,
                batch_size=64,
                optimizer="adamw",
                adversarial_eps=adv_eps,
                seed=40,
            ),
        )
        rows.append(
            {
                "experiment": "adversarial",
                "name": name,
                "clean_test_acc": _acc(model, data.X_test, data.y_test),
                "fgsm_test_acc": _adv_acc(model, data.X_test, data.y_test, eps),
                "train_acc": _acc(model, data.X_train, data.y_train),
            }
        )
    plot_adversarial(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="shorter epochs for a smoke run",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="subset: geometry linear mlp cnn bagging sparse adv",
    )
    args = parser.parse_args()
    epochs = 8 if args.quick else 25
    cnn_epochs = 6 if args.quick else 15
    wanted = set(args.only) if args.only else {
        "geometry",
        "linear",
        "mlp",
        "cnn",
        "bagging",
        "sparse",
        "adv",
    }
    # old aliases
    if "moons" in wanted:
        wanted.add("mlp")
    if "digits" in wanted:
        wanted.add("cnn")

    summary: list[dict] = []
    print(f"device=cpu  epochs={epochs}  results={RESULTS}")

    if "geometry" in wanted:
        print(">> geometry (closed form, no training)")
        exp_geometry()

    if "linear" in wanted:
        print(">> linear L1/L2")
        summary.extend(exp_linear())

    if "mlp" in wanted:
        print(">> mlp Fashion-MNIST")
        rows, _ = exp_mlp_digits(epochs)
        summary.extend(rows)

    if "cnn" in wanted:
        print(">> cnn vs mlp + aug")
        summary.extend(exp_cnn_and_aug(cnn_epochs))

    if "bagging" in wanted:
        print(">> bagging")
        summary.extend(exp_bagging(min(epochs, 12), n_models=3 if not args.quick else 2))

    if "sparse" in wanted:
        print(">> sparse hidden")
        summary.extend(exp_sparse_hidden(min(epochs, 15)))

    if "adv" in wanted:
        print(">> adversarial FGSM")
        summary.extend(exp_adversarial(cnn_epochs, eps=0.12))

    payload = _jsonable(summary)
    (RESULTS / "summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"wrote {RESULTS / 'summary.json'}")


if __name__ == "__main__":
    main()
