# Optimization experiments (Goodfellow ch. 8)

Notes: [NOTES.md](NOTES.md) — Unicode / ASCII formulas (Agents Window has no LaTeX). Pretty `$...$` math is in the notebook, Editor Window only.  
Chapter: https://www.deeplearningbook.org/contents/optimization.html

No generated data.

- **geometry**: analytic 2-D bowls (ill-conditioning, saddle, cliff). Not a fake dataset.
- **linear**: sklearn `diabetes`
- **vision**: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) — first run pulls ~30MB from Zalando into `data_cache/` (or reuses `regularization/data_cache/`)

```
python -m pip install -r requirements.txt
python test_algos.py
jupyter notebook optimization.ipynb   # Editor Window, not Agents
python run.py --quick
```

**Notebook:** open `optimization.ipynb` in Cursor's **Editor Window** (File → Open Editor Window). Python 3 kernel → **Shift+Enter**. Geometry is instant sliders. Fashion trains a small net (Ctrl+Enter after you change a knob). JSON view = you're in Agents Window.

## What the knobs are

| knob | section | plain meaning |
|---|---|---|
| `κ` | 8.2.1 | how skinny the bowl is |
| `ε` / `lr` | 8.3 | step size |
| `β` | 8.3 | how much past velocity you keep |
| `batch_size` | 8.1.3 | how noisy the gradient is |
| `clip` | 8.2.4 | cap on `‖g‖` |
| `damp` | 8.6 | Newton → SGD as `λ` grows |
| init | 8.4 | zero / tiny / xavier / he / huge |
| BN / Polyak | 8.7 | landscape + averaged weights |

## This run (CPU, 800 Fashion-MNIST train, 8 epochs, MLP 128–128)

| optimizer | train | test | gap |
|---|---|---|---|
| SGD `ε=0.05` | 0.76 | 0.71 | 0.05 |
| momentum `β=0.9` | 0.82 | **0.76** | 0.06 |
| Adam `ε=3e-3` | 0.86 | 0.77 | 0.09 |

Adam wins train; test is close to momentum. Faster `J` ≠ better `P` — that's §8.1. `zero` init stays at chance 0.10. Deep sigmoid: first-layer `|g|` ~ 1e-5 vs ReLU ~ 1e-3.
