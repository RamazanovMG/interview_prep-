# Regularization experiments (Goodfellow ch. 7)

Notes: [NOTES.md](NOTES.md)  
Chapter: https://www.deeplearningbook.org/contents/regularization.html

No generated data.

- **linear**: sklearn `diabetes`, `breast_cancer` (shipped with sklearn)
- **vision**: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) — first `run.py` pulls ~30MB from Zalando’s GitHub into `data_cache/` (gitignored)

```
python -m pip install -r requirements.txt
python test_closed_form.py
python run.py              # first run downloads Fashion-MNIST, then trains
python run.py --quick
python run.py --only geometry linear
```

`TrainConfig.weight_decay` hits weights only (ch. 7.1: leave biases alone).

## This run (CPU, 1500 Fashion-MNIST train)

256-256 MLP:

| regularizer | train | test | gap |
|---|---|---|---|
| none | 0.935 | 0.787 | 0.15 |
| L2 | 0.934 | 0.797 | 0.14 |
| L1 | 0.843 | 0.784 | 0.06 |
| dropout | 0.872 | 0.796 | 0.08 |
| input noise | 0.905 | 0.803 | 0.10 |
| **early stop** | 0.916 | **0.812** | 0.10 |
| label smooth | 0.957 | 0.798 | 0.16 |
| l2+dropout | 0.886 | 0.793 | 0.09 |

- CNN (20k params) test **0.857** vs MLP (269k) **0.791** — sharing wins
- FGSM: clean net 0.14 attacked → FGSM-train **0.59**, clean acc almost unchanged
- diabetes Lasso still edges OLS; cancer L1 keeps 6/30 features
