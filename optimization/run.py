"""Small Fashion bake-off so README / CI have numbers. Not the notebook."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load_fashion_split
from train import TrainConfig, make_mlp, metrics_for, train_classifier

RESULTS = Path(__file__).resolve().parent / "results"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    epochs = 3 if args.quick else 8
    n_train = 400 if args.quick else 800
    data = load_fashion_split(n_train=n_train, n_val=400, n_test=800)
    board = {}
    for name, kw in (
        ("sgd", {"optimizer": "sgd", "lr": 0.05}),
        ("momentum", {"optimizer": "momentum", "lr": 0.05, "momentum": 0.9}),
        ("adam", {"optimizer": "adam", "lr": 3e-3}),
    ):
        model = make_mlp(data.n_features, data.n_classes, hidden=(128, 128), seed=0)
        cfg = TrainConfig(epochs=epochs, batch_size=64, seed=0, **kw)
        train_classifier(model, data.X_train, data.y_train, data.X_val, data.y_val, cfg)
        met = metrics_for(
            model,
            data.X_train,
            data.y_train,
            data.X_val,
            data.y_val,
            data.X_test,
            data.y_test,
        )
        board[name] = {
            "train": round(met.train_acc, 4),
            "val": round(met.val_acc, 4),
            "test": round(met.test_acc, 4),
            "gap": round(met.gap, 4),
        }
        print(name, board[name])
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(board, indent=2) + "\n")
    print("wrote", RESULTS / "summary.json")


if __name__ == "__main__":
    main()
