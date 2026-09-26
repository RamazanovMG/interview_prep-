# Goodfellow ch. 7 — Regularization for Deep Learning

Source: https://www.deeplearningbook.org/contents/regularization.html

Agents Window / many markdown previews **do not run LaTeX**. Formulas here are Unicode + indented code so they stay readable. Pretty math lives in `regularization.ipynb` (open that in the **Editor Window**, not Agents).

Regularization = any change to a learning algorithm intended to **cut generalization error, not training error** (§5.2.2, restated in ch. 7).

The interesting DL fact: the best model is almost never “the right size.” It’s a **large model + the right regularizer**. Regularization buys a profitable bias–variance trade: more bias, a lot less variance.

Interview one-liner: *capacity control is not “fewer parameters.” It’s “which solutions in a huge hypothesis class you make cheap.”*

---

## 7.1 Parameter norm penalties

```
J̃(θ; X, y) = J(θ; X, y) + α Ω(θ)
```

- α = 0: no penalty. Larger α: more regularization.
- Penalize **weights, not biases**. A bias is one number per unit; shrinking it underfits for almost no variance win.
- Different α per layer is allowed; usually not worth the search.

### 7.1.1 L2 / weight decay / ridge / Tikhonov

```
Ω = (1/2) ‖w‖₂²

w  ←  (1 − εα) w  −  ε ∇_w J
```

Every SGD step **multiplicatively shrinks** `w` toward 0, then applies the data gradient. Gradient of the penalty is `α w`: proportional to the weight, **zero at zero**. That’s why L2 never lands on exact 0.

Quadratic analysis around the unregularized minimizer `w*`, Hessian `H = ∇²J(w*)`.
Rotate into the eigenbasis: `H = Q Λ Qᵀ`, and write the aligned OLS solution as `w*_aligned = Qᵀ w*`.

```
ŵ = (H + α I)⁻¹ H w*
  = Q (Λ + α I)⁻¹ Λ Qᵀ w*

ŵᵢ = [λᵢ / (λᵢ + α)] · (Qᵀ w*)ᵢ
```

- `λᵢ ≫ α`: data already pinned this direction → almost no shrink.
- `λᵢ ≪ α`: poorly determined (high variance) → smashed toward 0.

That’s Fig. 7.1. MAP view: Gaussian prior on `w`.

Linear least squares = ridge normal equation:

```
w = (Xᵀ X + α I)⁻¹ Xᵀ y
```

`closed_form.py` implements both. Figure: `results/fig71_l2_geometry.png`.

### 7.1.2 L1 / lasso

```
Ω = ‖w‖₁

∇ J̃ = α sign(w) + ∇J     (w ≠ 0)
```

Penalty gradient is **constant-magnitude**, not proportional to `w`. It does not fade as `w → 0`, so it can ride the weight onto the axis.

On a diagonal Hessian (uncorrelated / PCA’d features), soft-threshold (eq 7.23):

```
ŵᵢ = sign(w*ᵢ) · max( |w*ᵢ| − α / Hᵢᵢ ,  0 )
```

Dies exactly at 0 when `α ≥ |w*ᵢ| · Hᵢᵢ`. At `w = 0` the subdifferential of `|w|` is the interval `[-1, 1]`: if the data force `|H w*| ≤ α`, zero is a stationary point.

L2 never hits exact 0 if `w* ≠ 0`. L1 does → **feature selection**. MAP: Laplace prior.

Figures: `results/l1_vs_l2_shrinkage.png`, stems in `linear_weights.png`.

---

## 7.2 Penalties as constrained optimization

```
min J + α Ω     is the Lagrangian of     min J   s.t.   Ω(θ) ≤ k
```

α is the multiplier for some `k(α)`. Bigger α ⇔ smaller feasible set.

Geometry: L2 feasible set is a **disk** (optimum usually off-axis). L1 is a **diamond** (optimum often a **vertex** → zeros). `results/l1_l2_constraint_sets.png`.

Hard constraints (project back onto `‖w‖ ≤ k`) vs penalties: constraints stay inside a known radius even if a step would explode; useful when you want bounded weights for numerical reasons, not just a prior.

---

## 7.3 Under-constrained problems

More params than data → `XᵀX` singular, infinitely many interpolators. Adding `α I` makes it full rank. Moore–Penrose pseudoinverse is the `α → 0⁺` limit of ridge (min-`‖w‖` interpolator). Same reason weight decay saves you when two hidden units are co-adapted and `H` is degenerate.

---

## 7.4 Dataset augmentation

Manufacture new `(x, y)` by transforming `x` **without changing the label**.

- Images: translate, (sometimes) rotate/scale. Already-convolutional models still benefit.
- Do **not** flip `b`/`d` or rotate `6`/`9`.
- Input Gaussian noise is a domain-agnostic augmenter. Domain-specific warps are usually counted as preprocessing when you compare algorithms — don’t credit the architecture for the warps.

In the notebook: CNN + `±` pixel rolls (`MAX_SHIFT`, `N_COPIES`).

---

## 7.5 Noise robustness

- Infinitesimal input noise ≈ weight decay (Bishop). Finite / hidden-unit noise is **strictly more powerful** than shrinking `w`.
- Weight noise ≈ stochastic Bayesian inference over `w` (mostly RNNs in the book).
- **Dropout is hidden-unit multiplicative noise** (see 7.12).
- **§7.5.1 label smoothing**: assume the label is wrong with probability ε. Replace one-hot `y` with

```
(1 − ε) y + ε / K
```

Stops the net slamming softmax logits to `±∞`.

Notebook knobs: `NOISE`, `EPS`.

---

## 7.6 / 7.7 Semi-supervised and multitask

Not implemented here.

- Semi-supervised: unlabeled `x` informs `P(x)`, which (if `P(x)` and `P(y|x)` share structure) regularizes the classifier.
- Multitask: shared hidden trunk, task-specific heads. Extra tasks are a prior that the shared factors are “real.”

---

## 7.8 Early stopping

Most-used regularizer in DL because it’s free.

Train, track val, **keep the params at the best val epoch**, stop after `patience` non-improvements (Alg. 7.1). Val curve is U-shaped; training time is just another capacity knob, except you try every value of it in one run.

Bishop / Sjöberg–Ljung: under quadratic `J` + GD, early stopping ≡ L2. Number of steps ↔ `1/α`. Difference in practice: early stopping needs a val set and a stored snapshot; L2 doesn’t.

`none` keeps the **last** epoch (overfit). `early_stop` restores the snapshot. Knobs: `PATIENCE`, `MIN_EPOCH`.

---

## 7.9 Parameter tying / sharing

Tying: penalty `‖w_A − w_B‖²` (e.g. unsupervised pretrain vs classifier).

Sharing: **force equality**. CNN: one 3×3 kernel reused at every location — translation prior + huge memory win. Notebook: TinyCNN vs fat MLP on Fashion-MNIST.

---

## 7.10 Sparse representations

Two different sparsities:

| | what is zero | mechanism |
|---|---|---|
| sparse **parameters** | weights | L1 on `W` |
| sparse **representation** | hidden activations `h` | L1 (or KL / Student-t / ReLU) on `h` |

A dense `W` can still map `x` to a sparse `h`. Penalty is on `h`, which only indirectly shapes `W`. Knob: `ACT_L1`. Histogram: `activation_sparsity.png`.

---

## 7.11 Bagging and ensembles

Bootstrap the train set, train `k` models, average predictions. Each model overfits a different sample; the average cancels uncorrelated errors. Dropout is a cheap, implicit bag of `2ⁿ` subnetworks that share weights. Knob: `N_BAG`.

---

## 7.12 Dropout

At train time, multiply each hidden unit by Bernoulli(1 − p). PyTorch’s `Dropout` is **inverted**: it already rescales by `1/(1−p)` so eval is a no-op.

Interpretation:

1. Noise on hidden units / data aug at every layer.
2. Approximate geometric ensemble of subnets.
3. Breaks co-adaptation: a unit cannot rely on a particular collaborator being present.

Test-time weight scaling is the first-order approximation to the geometric mean of the ensemble. Works stupidly well with ReLU / maxout; p = 0.5 hidden, 0.2 input are the book’s defaults. Knob: `DROPOUT`.

---

## 7.13 Adversarial training

Nets that classify `x` correctly are often wrong on `x + ε` with ε tiny in L∞ and aligned with `∇_x J`. FGSM:

```
x_adv = x + ε · sign(∇_x J(x, y))
```

Training on those examples is a regularizer: it locally flattens the decision surface. Different from random input noise — the perturbation is **worst-case**, not isotropic. Knob: `ADV_EPS`.

---

## 7.14 Tangent prop / manifold tangent classifier

Prior: the decision function should be invariant along the data manifold’s tangent (small translations, etc.). Tangent prop penalizes `‖∇_x f · v_k‖` for known tangent vectors `v_k`. Augmentation is the Monte-Carlo version of the same prior. Not coded.

---

## Experiment map

| run | book | data | artifact |
|---|---|---|---|
| geometry | 7.1–7.2, eqs 7.13 / 7.23 | closed form | `fig71_l2_geometry.png`, `l1_l2_constraint_sets.png`, `l1_vs_l2_shrinkage.png` |
| linear | 7.1, 7.3 | sklearn diabetes + breast_cancer | `linear_weights.png`, `cancer_weights.png` |
| MLP regularizers | 7.1, 7.5, 7.8, 7.12 | Fashion-MNIST n=1500 | `mlp_boundaries.png`, `learning_curves.png`, `comparison_bars.png` |
| CNN / aug | 7.4, 7.9 | Fashion-MNIST | `digits_cnn_vs_mlp.png` |
| bagging | 7.11 | Fashion-MNIST | `bagging.png` |
| sparse hidden | 7.10 | Fashion-MNIST | `activation_sparsity.png` |
| FGSM | 7.13 | Fashion-MNIST | `adversarial.png` |

No synthetic samples. Linear: sklearn `load_diabetes` / `load_breast_cancer`. Vision: official Fashion-MNIST (auto-fetched from [Zalando](https://github.com/zalandoresearch/fashion-mnist) into `data_cache/`). Interactive walkthrough: `regularization.ipynb` in the Editor Window.

## Interview questions this folder is for

1. Why doesn’t L2 produce exact zeros? Write the 1-D shrink factor `λ / (λ + α)`.
2. Derive the SGD weight-decay update from `J̃ = J + (α/2) ‖w‖²`.
3. Why skip bias decay?
4. Early stopping ≡ L2 under what assumptions? What’s cheaper in practice?
5. Dropout train vs eval. Why inverted dropout?
6. Parameter sharing vs L2 tying: memory, inductive bias.
7. Random input noise vs FGSM: which prior?
8. “More parameters ⇒ more overfit” — when is that wrong?
