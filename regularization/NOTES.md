# Goodfellow ch. 7 — Regularization for Deep Learning

Source: https://www.deeplearningbook.org/contents/regularization.html

Regularization = any change to a learning algorithm intended to **cut generalization error, not training error** (§5.2.2, restated in ch. 7).

The interesting DL fact: the best model is almost never “the right size.” It’s a **large model + the right regularizer**. We are always fitting a square peg (the true process, which is “simulate the universe”) into a round hole (our architecture). Regularization is how you buy a profitable bias–variance trade: more bias, a lot less variance.

Interview one-liner: *capacity control is not “fewer parameters.” It’s “which solutions in a huge hypothesis class you make cheap.”*

---

## 7.1 Parameter norm penalties

\[
\tilde J(\theta; X, y) = J(\theta; X, y) + \alpha\,\Omega(\theta)
\]

- \(\alpha=0\): no penalty. Larger \(\alpha\): more regularization.
- Penalize **weights, not biases**. A bias is one number per unit; shrinking it underfits for almost no variance win.
- Different \(\alpha\) per layer is allowed; usually not worth the search.

### 7.1.1 \(L_2\) / weight decay / ridge / Tikhonov

\[
\Omega = \tfrac12 \|w\|_2^2, \qquad
w \leftarrow (1-\epsilon\alpha)\,w - \epsilon\nabla_w J
\]

Every SGD step **multiplicatively shrinks** \(w\) toward 0, then applies the data gradient.

Quadratic analysis around the unregularized minimizer \(w^*\), Hessian \(H=\nabla^2 J(w^*)\):

\[
\tilde w = (H + \alpha I)^{-1} H w^* = Q\,(\Lambda+\alpha I)^{-1}\Lambda\,Q^\top w^*
\]

Component \(i\) (eigenbasis of \(H\)) is rescaled by \(\lambda_i/(\lambda_i+\alpha)\):

- \(\lambda_i \gg \alpha\): data already pinned this direction down → almost no shrink.
- \(\lambda_i \ll \alpha\): direction is poorly determined (high variance) → smashed toward 0.

That’s Fig. 7.1. MAP view: Gaussian prior on \(w\).

For linear least squares this is the normal equation with a ridge:

\[
w = (X^\top X + \alpha I)^{-1} X^\top y
\]

`closed_form.py` implements both. `results/fig71_l2_geometry.png` is the figure.

### 7.1.2 \(L_1\) / lasso

\[
\Omega = \|w\|_1, \qquad
\nabla_w \tilde J = \alpha\,\mathrm{sign}(w) + \nabla_w J
\]

Penalty gradient is **constant-magnitude**, not proportional to \(w\). On a diagonal Hessian (uncorrelated / PCA’d features) you get soft-thresholding:

\[
\tilde w_i = \mathrm{sign}(w_i^*)\max\Bigl(|w_i^*| - \tfrac{\alpha}{H_{ii}},\,0\Bigr)
\]

\(L_2\) never hits exact 0 if \(w_i^*\neq 0\). \(L_1\) does, which is why people use it as **feature selection**. MAP view: Laplace prior.

See `results/l1_vs_l2_shrinkage.png` and the stem plots in `linear_weights.png`.

---

## 7.2 Penalties as constrained optimization

\(\tilde J = J + \alpha\Omega\) is the Lagrangian for \(\min J\) s.t. \(\Omega(\theta)\le k\). \(\alpha\) is the multiplier for some \(k(\alpha)\).

Geometry: \(L_2\) feasible set is a disk (optimum usually off-axis). \(L_1\) is a diamond (optimum often a **vertex** → zeros). `results/l1_l2_constraint_sets.png`.

Hard constraints (project back onto \(\|w\|\le k\)) vs penalties: constraints stay inside a known radius even if a step would explode; useful when you want bounded weights for numerical reasons, not just a prior.

---

## 7.3 Under-constrained problems

More params than data → \(X^\top X\) singular, infinitely many interpolators. Adding \(\alpha I\) makes it full rank. Moore–Penrose pseudoinverse is the \(\alpha\to 0^+\) limit of ridge (min-\(\|w\|\) interpolator). Same reason weight decay saves you when two hidden units are co-adapted and \(H\) is degenerate.

---

## 7.4 Dataset augmentation

Manufacture new \((x,y)\) by transforming \(x\) **without changing the label**.

- Images: translate, (sometimes) rotate/scale. Already-convolutional models still benefit.
- Do **not** flip `b`/`d` or rotate `6`/`9`.
- Input Gaussian noise is a domain-agnostic augmenter. Domain-specific warps are usually counted as preprocessing when you compare algorithms — don’t credit the architecture for the warps.

Digits experiment: CNN trained on \(\pm 1\) px rolls vs the same CNN without them.

---

## 7.5 Noise robustness

- Infinitesimal input noise \(\approx\) weight decay (Bishop). Finite / hidden-unit noise is **strictly more powerful** than shrinking \(w\).
- Weight noise ≈ stochastic Bayesian inference over \(w\) (mostly RNNs in the book).
- **Dropout is hidden-unit multiplicative noise** (see 7.12).
- **§7.5.1 output noise / label smoothing**: assume the label is wrong with probability \(\epsilon\). Replace one-hot \(y\) with \((1-\epsilon)y + \epsilon/k\). Stops the network from slamming softmax logits to \(\pm\infty\).

We run input-noise and `label_smoothing` as first-class moons configs.

---

## 7.6 / 7.7 Semi-supervised and multitask

Not implemented here.

- Semi-supervised: unlabeled \(x\) informs \(P(x)\), which (if \(P(x)\) and \(P(y|x)\) share structure) regularizes the classifier.
- Multitask: shared hidden trunk, task-specific heads. The extra tasks are a prior that the shared factors are “real.”

---

## 7.8 Early stopping

Most-used regularizer in DL because it’s free.

Train, track val, **keep the params at the best val epoch**, stop after `patience` non-improvements (Alg. 7.1). Val curve is U-shaped; training time is just another capacity knob, except you try every value of it in one run.

Bishop / Sjöberg–Ljung: under quadratic \(J\) + GD, early stopping \(\equiv L_2\). Number of steps \(\leftrightarrow 1/\alpha\). Difference in practice: early stopping needs a val set and a stored snapshot; \(L_2\) doesn’t.

`none` in the moons run keeps the **last** epoch (overfit). `early_stop` restores the snapshot. Learning curves mark the chosen epoch.

---

## 7.9 Parameter tying / sharing

Tying: penalty \(\|w_A - w_B\|^2\) (e.g. unsupervised pretrain vs classifier).

Sharing: **force equality**. CNN: one 3×3 kernel reused at every location — translation prior + huge memory win. That’s the experiment `cnn (param sharing)` vs a fat MLP on the same 8×8 digits.

---

## 7.10 Sparse representations

Two different sparsities:

| | what is zero | mechanism |
|---|---|---|
| sparse **parameters** | weights | \(L_1\) on \(w\) |
| sparse **representation** | hidden activations | \(L_1\) (or KL / Student-t / ReLU) on \(h\) |

A dense \(W\) can still map \(x\) to a sparse \(h\). Penalty is on \(h\), which only indirectly shapes \(W\). Histogram: `activation_sparsity.png`.

---

## 7.11 Bagging and ensembles

Bootstrap the train set, train \(k\) models, average predictions. Each model overfits a different sample; the average cancels uncorrelated errors. Dropout is a cheap, implicit bag of \(2^n\) subnetworks that share weights.

---

## 7.12 Dropout

At train time, multiply each hidden unit by \(\mathrm{Bernoulli}(1-p)\) (PyTorch’s `Dropout` is inverted: it already rescales by \(1/(1-p)\) so eval is a no-op).

Interpretation:

1. Noise on hidden units / data aug at every layer.
2. Approximate geometric ensemble of subnets.
3. Breaks co-adaptation: a unit cannot rely on a particular collaborator being present.

Test-time weight scaling is the first-order approximation to the geometric mean of the ensemble. Works stupidly well with ReLU / maxout; \(p=0.5\) hidden, \(0.2\) input are the book’s defaults.

---

## 7.13 Adversarial training

Nets that classify \(x\) correctly are often wrong on \(x+\varepsilon\) with \(\varepsilon\) tiny in \(L_\infty\) and aligned with \(\nabla_x J\). FGSM:

\[
x_{\mathrm{adv}} = x + \varepsilon\,\mathrm{sign}(\nabla_x J(x,y))
\]

Training on those examples is a regularizer: it locally flattens the decision surface. Different from random input noise — the perturbation is **worst-case**, not isotropic.

---

## 7.14 Tangent prop / manifold tangent classifier

Prior: the decision function should be invariant along the data manifold’s tangent (small translations, etc.). Tangent prop penalizes \(\|\nabla_x f \cdot v_k\|\) for known tangent vectors \(v_k\). Augmentation is the Monte-Carlo version of the same prior. Not coded.

---

## Experiment map

| run | book | data | artifact |
|---|---|---|---|
| geometry | 7.1–7.2, eqs 7.13 / 7.23 | closed form | `fig71_l2_geometry.png`, `l1_l2_constraint_sets.png`, `l1_vs_l2_shrinkage.png` |
| linear | 7.1, 7.3 | sklearn diabetes + breast_cancer | `linear_weights.png`, `cancer_weights.png` |
| MLP regularizers | 7.1, 7.5, 7.8, 7.12 | sklearn digits 10-way, n=80 | `mlp_boundaries.png`, `learning_curves.png`, `comparison_bars.png` |
| CNN / aug | 7.4, 7.9 | sklearn digits 10-way | `digits_cnn_vs_mlp.png` |
| bagging | 7.11 | digits 10-way | `bagging.png` |
| sparse hidden | 7.10 | digits 10-way | `activation_sparsity.png` |
| FGSM | 7.13 | digits 10-way | `adversarial.png` |

No synthetic samples. `sklearn.datasets` only (`load_diabetes`, `load_breast_cancer`, `load_digits`).

## Interview questions this folder is for

1. Why doesn’t \(L_2\) produce exact zeros? Write the 1-D shrink factor.
2. Derive the SGD weight-decay update from \(\tilde J = J + \frac{\alpha}{2}\|w\|^2\).
3. Why skip bias decay?
4. Early stopping \(\equiv L_2\) under what assumptions? What’s cheaper in practice?
5. Dropout train vs eval. Why inverted dropout?
6. Parameter sharing vs \(L_2\) tying: memory, inductive bias.
7. Random input noise vs FGSM: which prior?
8. “More parameters ⇒ more overfit” — when is that wrong?
