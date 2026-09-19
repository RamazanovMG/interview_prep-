# Regularization experiments (Goodfellow ch. 7)

Notes: [NOTES.md](NOTES.md)  
Chapter: https://www.deeplearningbook.org/contents/regularization.html

Same wide models, different regularizers. Datasets are tiny on purpose so an unregularized net can memorize.

```
python -m pip install -r requirements.txt
python test_closed_form.py
python run.py              # full suite, CPU, a few minutes
python run.py --quick      # shorter epochs
python run.py --only geometry linear
```

Writes `results/*.png` and `results/summary.json`.

| script | what |
|---|---|
| `closed_form.py` | eqs 7.13 / 7.17 / 7.23 |
| `data.py` | moons (overfit), high-p regression, sklearn digits |
| `models.py` | MLP, TinyCNN, FGSM, decay on weights only |
| `train.py` | L2 / L1 / act-L1 / dropout / input noise / label smooth / early stop / bagging |
| `run.py` | all experiments |

`TrainConfig.weight_decay` is applied only to `ndim>1` tensors (ch. 7.1: leave biases alone).
