# Goodfellow ch. 8 — Optimization for Training Deep Models

Source: https://www.deeplearningbook.org/contents/optimization.html

Agents Window / many markdown previews **do not run LaTeX**. Formulas here are Unicode + indented code so they stay readable. Pretty math lives in `optimization.ipynb` (open that in the **Editor Window**, not Agents).

Training a net is an optimization problem, but **not a pure one**. We care about test performance `P`. We actually minimize a stand-in `J(θ)` (average train loss, maybe plus a regularizer) and hope `P` follows.

Interview one-liner: *learning is risk minimization with a surrogate, a sample, and a noisy gradient. The optimizer is how you walk that landscape.*

---

## 8.1 Learning differs from pure optimization

```
J(θ)  = E_{(x,y) ~ p̂_data}  L(f(x; θ), y)     empirical risk   (8.1)
J*(θ) = E_{(x,y) ~ p_data}   L(f(x; θ), y)     true risk        (8.2)
```

Pure optimization: `J` is the goal. ML: `J*` is the goal, `J` is what we can compute. A smaller train loss is useful only if it tracks the test.

### 8.1.1 / 8.1.2 Empirical risk and surrogates

0-1 loss is what you want for classification and is awful to optimize (flat, discontinuous). We minimize a **smooth surrogate** (cross-entropy, hinge). The surrogate can go to 0 while 0-1 is already perfect — that's why train CE keeps falling after train acc saturates.

### 8.1.3 Minibatch

Exact gradient = average over the whole train set. A minibatch of size `m` is an unbiased (noisy) estimate. Cost per step is `O(m)`, noise falls like `1/√m`.

- `m = n` (full batch): smooth path, expensive, easy to overfit the exact train gradient.
- `m = 1`: cheap, very noisy, can bounce out of sharp holes.
- Sweet spot is a power of two that fills the hardware (32–256 here).

Notebook knob: `batch_size` on diabetes (instant) and Fashion.

---

## 8.2 Challenges

### 8.2.1 Ill-conditioning

Hessian `H` has eigenvalues `λ`. Condition number `κ = λ_max / λ_min`.

```
SGD on a quadratic is stable only for   ε < 2 / λ_max
```

The long, skinny axis (`λ_min` tiny) needs a **big** step. The short axis forbids it. SGD zigzags: each step is either too timid along the valley or too wild across it. That's Fig. 8.2.

### 8.2.2 Local minima

In high-dim nets a "bad" local min with high cost is rare. Most critical points that trap you are **saddles**: min along some axes, max along others.

### 8.2.3 Plateaus, saddles, flat regions

```
J = (x² − y²) / 2     saddle at 0
```

On the ridge `y = 0` the gradient is 0. SGD started exactly there stays forever. A tiny bump, or noise from a minibatch, rolls off. That's why SGD noise is not only a bug.

### 8.2.4 Cliffs and exploding gradients

A steep wall: `‖∇J‖` huge, one step jumps across the whole map (or to NaN). **Gradient clipping** caps `‖g‖` and keeps the direction:

```
if ‖g‖ > c:   g ← g · (c / ‖g‖)
```

This is a step-size fix, not a new direction.

### 8.2.5 Long-term dependencies / vanishing

A chain of Jacobians. If each multiply is `< 1`, the product → 0 (tanh/sigmoid). If `> 1`, it explodes. Deep tanh: bottom layers get a dead gradient. ReLU / residual / BN are architecture answers; clipping is the exploding answer.

### 8.2.6 Inexact gradients

Minibatch, dropout, data aug — the g you follow is wrong. Adaptive methods and smaller `ε` tolerate that. Newton's exact `H⁻¹g` does not: a noisy `H` is worse than a noisy `g`.

---

## 8.3 Basic algorithms

```
SGD:        θ ← θ − ε g

momentum:   v ← β v − ε g
            θ ← θ + v

Nesterov:   g is evaluated at θ + β v   (look ahead, then correct)
```

`β ≈ 0.9` means "keep ~10 steps of history". Momentum averages the zigzag and builds speed along the valley. Too big `β` + too big `ε` → overshoot.

Nesterov sees the cliff before it jumps.

---

## 8.4 Initialization

All-zero (or all-equal) weights: hidden units stay **symmetric** forever. Gradient is identical, they never specialize.

Need to break symmetry **and** keep activations / grads from vanishing or exploding on the first pass.

```
Xavier / Glorot:   Var(W) ≈ 2 / (n_in + n_out)     (tanh / linear)
He / Kaiming:      Var(W) ≈ 2 / n_in               (ReLU)
```

Tiny init: signal dies before the last layer. Huge init: activations saturate / loss is NaN.

Notebook: `zero` / `tiny` / `xavier` / `he` / `huge`.

---

## 8.5 Adaptive learning rates

One global `ε` is a compromise across coordinates (see ill-conditioning). Adaptive methods keep a per-weight step.

```
AdaGrad:   acc ← acc + g²
           θ  ← θ − ε g / √acc
```

`acc` only grows. Frequent features get a tiny step and eventually freeze. Good for sparse, bad for long training.

```
RMSProp:   s ← ρ s + (1−ρ) g²
           θ ← θ − ε g / √s
```

Leaky AdaGrad. Forgets old g². Default `ρ = 0.9`.

```
Adam:      m ← β1 m + (1−β1) g          (momentum)
           v ← β2 v + (1−β2) g²         (RMSProp)
           then bias-correct m̂, v̂ because they start at 0
           θ ← θ − ε m̂ / √v̂
```

Defaults `β1=0.9`, `β2=0.999`, `ε=1e-3` (Adam's ε, not SGD's). First-week default for nets. Still not magic: too big `ε` diverges, weight decay should be AdamW not "L2 inside Adam".

---

## 8.6 Approximate second-order

```
Newton:   θ ← θ − H⁻¹ ∇J
```

On a quadratic, **one** step is the exact minimum. On diabetes linear MSE that's just OLS.

Why not ImageNet:

- `H` is `p × p`. For a 1e6-weight net that's impossible to store.
- `H` is not even positive-definite near saddles → Newton jumps the **wrong** way (up the hill).
- Noisy minibatch `H` is garbage.

Damping (`H + λ I`, Levenberg) makes it PD and shrinks the step toward GD. `λ → ∞` → SGD. L-BFGS / CG approximate `H⁻¹` with a few vectors; still rare for large nets.

Notebook: Newton vs SGD on sklearn diabetes.

---

## 8.7 Strategies / meta-algorithms

**Batch norm (§8.7.1).** Normalize each hidden coord over the minibatch, then learn scale/shift. Makes the landscape less ill-conditioned, lets you raise `ε`, less sensitive to init. At eval: use running mean/var, not the batch.

**Coordinate descent.** Optimize one block at a time. Rare for dense nets; the book mentions it for sparse / classic models.

**Polyak averaging (§8.7.3).**

```
θ̄ ← (t/(t+1)) θ̄ + (1/(t+1)) θ
```

The running mean of the iterates. Oscillations cancel. Often better test than the last noisy `θ`. Cheap: one extra copy of the weights.

**Greedy supervised pretraining.** Train a shallow piece, freeze, stack. Historical. Residual / BN / Adam mostly replaced it.

**Design the model so optimization is easy.** ReLU, skip connections, BN — these are optimization tools, not just "architecture taste".

**Curriculum / continuation.** Start with an easy loss / easy examples, slowly harden. Homotopy: solve a sequence of problems that deform into the one you want.

---

## Experiment map

| run | book | data | what you turn |
|---|---|---|---|
| ill-conditioned bowl | 8.2.1 | closed-form quadratic | `κ`, `ε` |
| momentum / Nesterov / Adam | 8.3, 8.5 | same bowl | `ε`, `β` |
| saddle | 8.2.3 | `x² − y²` | `bump`, `ε` |
| cliff + clip | 8.2.4 | exponential wall | `clip` |
| minibatch | 8.1.3 | sklearn diabetes | `batch_size` |
| Newton | 8.6 | diabetes | `damp` |
| Fashion bake-off | 8.3–8.5 | Fashion-MNIST | optimizer, `ε`, batch |
| init | 8.4 | Fashion-MNIST | `zero/tiny/xavier/he/huge` |
| BN | 8.7.1 | Fashion-MNIST | on/off |
| Polyak | 8.7.3 | Fashion-MNIST | on/off |
| vanishing grads | 8.2.5 | Fashion-MNIST | `tanh` vs `relu`, depth |

No synthetic samples for the models. Geometry bowls are analytic plots, not fake datasets. Linear: sklearn `load_diabetes`. Vision: official Fashion-MNIST (same Zalando dump as `regularization/`, shared cache if present).

## Interview questions this folder is for

1. Why is training a net not the same as "just minimize J"? What do we actually care about?
2. Why minibatches? What changes when the batch is 1 vs the whole train set?
3. What is ill-conditioning? Why does SGD zigzag? What is the largest safe learning rate on a quadratic?
4. Why are saddle points a bigger deal than bad local minima in deep nets?
5. Write the SGD and momentum updates. What does the momentum coefficient actually do?
6. Why look-ahead (Nesterov) instead of classical momentum?
7. Why not initialize every weight to 0? What is Xavier/Glorot trying to keep constant?
8. AdaGrad vs RMSProp vs Adam — what problem does each one fix?
9. Why don't we Newton-train a big net? What does damping (`H + λI`) do?
10. What does batch norm stabilize, and why can you raise the learning rate after you add it?
11. Gradient clipping: vanishing or exploding? Does it change the direction or only the length?
12. Polyak averaging: what do you average, and why is the average often better than the last point?
