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
