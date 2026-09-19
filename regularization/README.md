# Regularization experiments (Goodfellow ch. 7)

Notes: [NOTES.md](NOTES.md)  
Chapter: https://www.deeplearningbook.org/contents/regularization.html

Same models, different regularizers. Data is **sklearn built-ins only** — diabetes, breast_cancer, digits. Nothing is generated, nothing is downloaded.

```
python -m pip install -r requirements.txt
python test_closed_form.py
python run.py              # full suite, CPU, ~15s
python run.py --quick      # shorter epochs
python run.py --only geometry linear
```

Writes `results/*.png` and `results/summary.json`.

| script | what |
|---|---|
| `closed_form.py` | eqs 7.13 / 7.17 / 7.23 |
| `data.py` | diabetes, breast_cancer, digits |
| `models.py` | MLP, TinyCNN, FGSM, decay on weights only |
| `train.py` | L2 / L1 / act-L1 / dropout / input noise / label smooth / early stop / bagging |
| `run.py` | all experiments |

`TrainConfig.weight_decay` is applied only to `ndim>1` tensors (ch. 7.1: leave biases alone).

Tiny train splits (`n=40–80`) so an unregularized net can actually memorize.

## What the CPU run showed

256-256 MLP, sklearn digits 10-way, **n_train=80** (train acc = 1.0 for almost every run):

| regularizer | test | gap |
|---|---|---|
| none | 0.829 | 0.17 |
| L2 | 0.856 | 0.14 |
| L1 | 0.849 | 0.15 |
| dropout | 0.836 | 0.16 |
| input noise | 0.860 | 0.14 |
| early stop | 0.847 | 0.15 |
| **label smooth** | **0.866** | **0.13** |
| l2+dropout | 0.848 | 0.15 |

Other hits (all sklearn data):

- **diabetes**: Lasso test MSE 3178 vs OLS 3238; zeros `s1`/`s4`-ish, keeps `bmi` + `s5`
- **breast_cancer**: L2 0.971 > none 0.967; L1 keeps 6/30 weights (worst concave points dominates)
- **digits CNN**: 3.8k params vs MLP 85k; `+1px` aug test 0.957
- **bagging x5**: 0.857 → 0.880
- **L1 on tanh h**: mean \|h\| 0.81 → 0.37
- **FGSM**: clean net 0.95 → 0.26 attacked; FGSM-train 0.95 / 0.59

Plots: `comparison_bars.png`, `cancer_weights.png`, `learning_curves.png`, `adversarial.png`
