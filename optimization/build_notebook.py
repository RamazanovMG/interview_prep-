"""Generate optimization.ipynb. Run: python build_notebook.py"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
NB = HERE / "optimization.ipynb"


def md(src: str):
    return nbf.v4.new_markdown_cell(src)


def code(src: str):
    return nbf.v4.new_code_cell(src)


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
    }
    cells = []

    cells.append(
        md(
            r"""# Optimization, cell by cell

Goodfellow et al., [ch. 8](https://www.deeplearningbook.org/contents/optimization.html). Конспект: `NOTES.md`.

**Обучение сети** — это оптимизация, но не «просто минимизируй $J$». Нам важна ошибка на тесте. Мы крутим другую штуку: среднюю ошибку на трейне (иногда плюс штраф) и надеемся, что тест поедет следом.

$$
J(\theta)=\mathbb{E}_{(x,y)\sim\hat p_{\mathrm{data}}}L(f(x;\theta),y)
$$

$J$ — то, что можем посчитать. Настоящая цель — то же матожидание по *настоящему* распределению, его нет в компьютере.

## Как идти

1. **Editor Window**, не Agents. Kernel Python 3 (`pip install -r requirements.txt`, нужен `ipywidgets`).
2. Setup один раз. **Shift+Enter** = ячейка + следующая.
3. Геометрия: **слайдеры**, тащи сразу. Fashion учит сеть — поменял крутилку, **Ctrl+Enter**.
4. Не жми Run All в первый раз.

| § | принцип | что крутить |
|---|---|---|
| 2 | плохо обусловленная чаша | $\kappa$, $\varepsilon$ |
| 3 | SGD / momentum / Nesterov / Adam | $\varepsilon$, $\beta$ |
| 4 | седло | bump |
| 5 | обрыв + clip | clip |
| 6 | размер батча | batch |
| 7 | Ньютон vs SGD | damp |
| 8–12 | Fashion: оптимайзер, init, BN, Polyak | крутилки в ячейке |
"""
        )
    )

    cells.append(
        md(
            "## 0. Setup\n\nОдин раз за kernel. Геометрия от `EPOCHS` / `N_TRAIN` не зависит.\n"
        )
    )

    cells.append(
        code(
            r"""%matplotlib inline
%config InlineBackend.figure_format = "retina"

import os, sys
from pathlib import Path

HERE = Path.cwd().resolve()
if (HERE / "optimization" / "train.py").exists():
    HERE = HERE / "optimization"
elif not (HERE / "train.py").exists():
    raise FileNotFoundError(f"can't find train.py from {Path.cwd()}")
os.chdir(HERE)
sys.path.insert(0, str(HERE))
print("cwd:", HERE)

import matplotlib.pyplot as plt
import numpy as np
import torch

from data import load_diabetes_split, load_fashion_split
from train import (
    TrainConfig,
    make_mlp,
    metrics_for,
    one_batch_grad_profile,
    train_classifier,
)
import viz

print("sliders:", "on" if viz.HAS_WIDGETS else "OFF — pip install ipywidgets, restart kernel")

plt.rcParams.update({"figure.figsize": (6.2, 4.0), "figure.dpi": 110})

EPOCHS = 6
N_TRAIN = 600
N_VAL = 500
N_TEST = 1000
HIDDEN = (128, 128)
SEED = 0

BOARD: dict[str, dict] = {}
HISTS: dict = {}
DATA = None


def plot_hist(hist, title: str = "") -> None:
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.3))
    ax[0].plot(hist.train_acc, label="train")
    ax[0].plot(hist.val_acc, label="val")
    ax[0].set_title("accuracy")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[1].plot(hist.train_loss, label="train")
    ax[1].plot(hist.val_loss, label="val")
    ax[1].set_title("loss")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    fig.suptitle(title, y=1.02)
    plt.show()


def fit(name: str, **cfg) -> tuple:
    global DATA
    if DATA is None:
        DATA = load_fashion_split(n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST, seed=SEED)
        print("Fashion", DATA.X_train.shape, "classes", DATA.n_classes)
    data = DATA
    model = make_mlp(
        data.n_features,
        data.n_classes,
        hidden=cfg.pop("hidden", HIDDEN),
        activation=cfg.pop("activation", "relu"),
        batch_norm=cfg.pop("batch_norm", False),
        init=cfg.pop("init", "xavier"),
        seed=cfg.get("seed", SEED),
    )
    tc = TrainConfig(
        epochs=cfg.get("epochs", EPOCHS),
        lr=cfg.get("lr", 3e-3),
        batch_size=cfg.get("batch_size", 64),
        optimizer=cfg.get("optimizer", "sgd"),
        momentum=cfg.get("momentum", 0.9),
        grad_clip=cfg.get("grad_clip"),
        polyak=cfg.get("polyak", False),
        seed=cfg.get("seed", SEED),
    )
    hist = train_classifier(model, data.X_train, data.y_train, data.X_val, data.y_val, tc)
    met = metrics_for(
        model, data.X_train, data.y_train, data.X_val, data.y_val, data.X_test, data.y_test
    )
    BOARD[name] = {"train": met.train_acc, "val": met.val_acc, "test": met.test_acc, "gap": met.gap}
    HISTS[name] = hist
    print(
        f"{name:16} train={met.train_acc:.3f}  val={met.val_acc:.3f}  "
        f"test={met.test_acc:.3f}  gap={met.gap:.3f}"
    )
    plot_hist(hist, name)
    return model, hist, met
"""
        )
    )

    cells.append(
        md(
            r"""## 1. Зачем это вообще (ch. 8.1)

Чистая оптимизация: $J$ и есть цель. В обучении цель — ошибка на *новых* картинках. $J$ на трейне — подделка, которую мы умеем считать.

Ещё подделка: 0-1 лосс (угадал класс или нет) почти нельзя дифференцировать. Берём гладкую замену — кросс-энтропию. Она может падать уже после того, как точность на трейне упёрлась в потолок. Это нормально, не баг.

Интервью-строка: *оптимизатор — это как ты ходишь по ландшафту подделки, надеясь прийти туда, где тест тоже хороший.*
"""
        )
    )

    cells.append(
        md(
            r"""## 2. Плохая обусловленность — почему SGD зигзаг (ch. 8.2.1)

Чаша вытянутая: вдоль одной оси ошибка меняется сильно, вдоль другой — почти нет.

$$
\kappa=\lambda_{\max}/\lambda_{\min},\qquad \varepsilon < 2/\lambda_{\max}
$$

$\kappa$ большое = чаша как блин. Большой шаг нужен вдоль длинной оси, но короткий шаг уже вышибает поперёк. SGD прыгает туда-сюда.

| крутишь | что происходит |
|---|---|
| $\kappa$ ↑ | чаша тоньше, зигзаг злее, до минимума дальше |
| $\varepsilon$ чуть ниже $2/\lambda_{\max}$ | доезжает, медленно по длинной оси |
| $\varepsilon$ выше порога | улетает. В печати будет «расходится» |

Синяя звезда — минимум. Серый путь — SGD. Сравни $\varepsilon=0.08$ и $\varepsilon=0.9$ при $\kappa=25$.
"""
        )
    )

    cells.append(
        code(
            """viz.play(
    viz.ill_conditioned,
    kappa=viz.fslider(25.0, min=1.0, max=80.0, step=1.0, description="κ  вытянутость"),
    lr=viz.fslider(0.08, min=0.01, max=1.2, step=0.01, description="ε  шаг"),
    steps=viz.fslider(40, min=5, max=120, step=1, description="шаги"),
)
"""
        )
    )

    cells.append(
        md(
            r"""## 3. SGD, momentum, Nesterov, Adam — один и тот же ландшафт (ch. 8.3 / 8.5)

$$
\text{SGD: } \theta\leftarrow\theta-\varepsilon g
\qquad
\text{momentum: } v\leftarrow\beta v-\varepsilon g,\;\theta\leftarrow\theta+v
$$

Momentum помнит прошлые шаги. Зигзаг поперёк чаши усредняется, вдоль долины набирается скорость.

Nesterov смотрит *куда уже несёт* и считает градиент там — меньше перехлёст.

Adam ещё и делит шаг на «насколько эта координата обычно дёргается». На вытянутой чаше ему можно тот же $\varepsilon$, что SGD убивает.

| крутишь | что происходит |
|---|---|
| $\varepsilon$ ↑ | все быстрее; SGD первый улетает |
| $\beta$ ↑ | momentum/Nesterov инерционнее, могут промахнуться мимо минимума |
| $\kappa$ ↑ | разрыв SGD vs остальные растёт |

Смотри $J_{\mathrm{end}}$ в печати. Меньше — ближе к дну.
"""
        )
    )

    cells.append(
        code(
            """viz.play(
    viz.first_order_bowl,
    lr=viz.fslider(0.08, min=0.01, max=0.4, step=0.01, description="ε  шаг"),
    beta=viz.fslider(0.85, min=0.0, max=0.99, step=0.01, description="β  инерция"),
    kappa=viz.fslider(18.0, min=1.0, max=60.0, step=1.0, description="κ"),
    steps=viz.fslider(50, min=10, max=120, step=1, description="шаги"),
)
"""
        )
    )

    cells.append(
        md(
            r"""## 4. Седло, не «плохой локальный минимум» (ch. 8.2.2–8.2.3)

В большой сети почти нет ловушек вида «яма с плохим значением». Есть сёдла: по одним осям минимум, по другим — горка.

$$
J=(x^2-y^2)/2
$$

На гребне $y=0$ градиент ноль. Если старт ровно там, SGD стоит вечно. Чуть сдвинь по $y$ — скатывается *вниз по горе*, $|J|$ растёт, это ок: седло не яма.

Шум минибатча = бесплатный bump. Поэтому шум градиента не только вред.

| крутишь | что происходит |
|---|---|
| bump $=0$ | стоит на седле |
| bump чуть > 0 | уезжает по $y$ |
| $\varepsilon$ ↑ | быстрее скатывается |
"""
        )
    )

    cells.append(
        code(
            """viz.play(
    viz.saddle_escape,
    lr=viz.fslider(0.15, min=0.02, max=0.5, step=0.01, description="ε"),
    bump=viz.fslider(0.04, min=0.0, max=0.4, step=0.01, description="bump по y"),
    steps=viz.fslider(40, min=5, max=80, step=1, description="шаги"),
)
"""
        )
    )

    cells.append(
        md(
            r"""## 5. Обрыв и клип градиента (ch. 8.2.4)

Стенка почти вертикальная: градиент огромный, один шаг — на другую планету (или NaN).

Клип: если длина $g$ больше порога, сократи длину, направление не трогай.

Это не «другой оптимайзер». Это предохранитель на размер шага.

| `clip` | что видишь |
|---|---|
| 0 (выкл) | серый путь перепрыгивает стенку на другую сторону |
| 1.5 | чёрный остаётся в чаше |
| 0.3 | слишком робко, ползёт |

Старт на стенке. Смотри `‖g‖ at start` — без клипа один шаг уносит по y.
"""
        )
    )

    cells.append(
        code(
            """viz.play(
    viz.cliff_clip,
    lr=viz.fslider(0.15, min=0.02, max=0.4, step=0.01, description="ε"),
    clip=viz.fslider(1.5, min=0.0, max=20.0, step=0.5, description="clip  0=выкл"),
    steps=viz.fslider(18, min=5, max=40, step=1, description="шаги"),
)
"""
        )
    )

    cells.append(
        md(
            r"""## 6. Минибатч — шумный, но честный градиент (ch. 8.1.3)

Полный градиент = среднее по всему трейну. Батч размера $m$ — несмещённая оценка, шум $\sim 1/\sqrt{m}$.

| batch | путь |
|---|---|
| весь трейн | гладкий, каждый шаг дорогой |
| 16–64 | обычный компромисс |
| 1 | дёшево, сильно трясёт, может выпрыгнуть из острой ямы |

Одна и та же линейная модель на sklearn diabetes (реальные признаки, не синтетика). Серая кривая — batch=1, синяя — полный батч, красная — твоя крутилка.

`lr` слишком большой на полном батче — тоже разъедется. Сначала подвигай только batch.
"""
        )
    )

    cells.append(
        code(
            """viz.play(
    viz.minibatch_diabetes,
    batch_size=viz.islider(16, min=1, max=200, step=1, description="batch"),
    lr=viz.fslider(0.08, min=0.005, max=0.4, step=0.005, description="ε", continuous_update=False),
    epochs=viz.islider(40, min=10, max=80, step=5, description="epochs"),
)
"""
        )
    )

    cells.append(
        md(
            r"""## 7. Ньютон: один шаг и ты на дне. Почему не ImageNet (ch. 8.6)

$$
\theta\leftarrow\theta-(H+\lambda I)^{-1}\nabla J
$$

На квадратике (линейная MSE) один шаг Ньютона = точное решение, обычный МНК.

Почему так не учат большие сети:

- $H$ размера «число весов × число весов» — не влезает
- в седле $H$ не положительно определён: Ньютон прыгает *в горку*
- шумный $H$ с минибатча хуже шумного градиента

`damp` = $\lambda$. Большая $\lambda$ превращает Ньютон в обычный градиентный шаг.

Горизонталь на графике — куда Ньютон пришёл сразу. Кривая — SGD 80 эпох.
"""
        )
    )

    cells.append(
        code(
            """viz.play(
    viz.newton_vs_sgd_diabetes,
    damp=viz.fslider(0.0, min=0.0, max=20.0, step=0.5, description="λ  damping"),
    sgd_lr=viz.fslider(0.05, min=0.005, max=0.2, step=0.005, description="ε SGD"),
)
"""
        )
    )

    cells.append(
        md(
            r"""## 8. Fashion: SGD с нормальным шагом (ch. 8.3)

Дальше — настоящие картинки, MLP 128–128. Сначала тупой SGD.

`lr=0.05` здесь ок. `lr=0.5` часто взрывается. `lr=0.001` ползёт.

Поменял `LR_SGD` → Ctrl+Enter. Кривые train/val: если train уехал вверх, а val нет — учится; если обе мёртвые — шаг не тот.
"""
        )
    )

    cells.append(
        code(
            """LR_SGD = 0.05
fit("sgd", optimizer="sgd", lr=LR_SGD)
"""
        )
    )

    cells.append(
        md(
            r"""## 9. Тот же Fashion, momentum (ch. 8.3)

Тот же `lr`, плюс память шагов `β=0.9`. Обычно val выше и раньше, чем у голого SGD.

Если кривая качается — урони `lr` или `β`. Если как SGD — `β` слишком маленький (0 = обычный SGD).
"""
        )
    )

    cells.append(
        code(
            """LR_MOM = 0.05
BETA = 0.9
fit("momentum", optimizer="momentum", lr=LR_MOM, momentum=BETA)
"""
        )
    )

    cells.append(
        md(
            r"""## 10. Adam (ch. 8.5.3)

Другой масштаб шага: дефолт `3e-3`, не `0.05`. Adam сам делит координаты — тот же `0.05` часто слишком злой.

Сравни test с `sgd` / `momentum` на табло в конце. На этой маленькой выборке Adam часто быстрее на трейне, тест не всегда лучший — это тоже урок главы: быстрее $J$ ≠ лучше $P$.
"""
        )
    )

    cells.append(
        code(
            """LR_ADAM = 3e-3
fit("adam", optimizer="adam", lr=LR_ADAM)
"""
        )
    )

    cells.append(
        md(
            r"""## 11. Размер батча на Fashion (ch. 8.1.3)

Тот же Adam, разные $m$. Мелкий батч шумнее и делает больше шагов на эпоху (у нас размер трейна фиксирован).

| batch | ожидай |
|---|---|
| 16 | шумные кривые, часто нормальный тест |
| 64 | дефолт |
| 256 | глаже, меньше шагов за эпоху, может недоучиться за те же EPOCHS |
"""
        )
    )

    cells.append(
        code(
            """BATCH = 16
fit("adam-b16", optimizer="adam", lr=3e-3, batch_size=BATCH)
"""
        )
    )

    cells.append(
        md(
            r"""## 12. Инициализация (ch. 8.4)

`zero` — все скрытые нейроны одинаковые навсегда. Сеть как один нейрон.

`tiny` — сигнал до последнего слоя дохнет.

`xavier` / `he` — нормальный старт (he чуть лучше для ReLU).

`huge` — активации взрываются, лосс NaN или плоский.

Смени `INIT`, Ctrl+Enter. Сравни train с `adam` выше.
"""
        )
    )

    cells.append(
        code(
            """INIT = "zero"   # zero | tiny | xavier | he | huge
fit(f"init-{INIT}", optimizer="adam", lr=3e-3, init=INIT)
"""
        )
    )

    cells.append(
        md(
            r"""## 13. Batch norm (ch. 8.7.1)

По батчу нормализуем каждую скрытую координату, потом учим свой масштаб и сдвиг.

Ландшафт становится менее «блином». Можно более злой `lr`. Меньше зависимость от init.

На тесте BN использует бегущее среднее, не текущий батч — поэтому `model.eval()` обязателен (у нас так в `evaluate`).
"""
        )
    )

    cells.append(
        code(
            """fit("adam+bn", optimizer="adam", lr=1e-2, batch_norm=True)
"""
        )
    )

    cells.append(
        md(
            r"""## 14. Polyak averaging (ch. 8.7.3)

После каждого шага держим среднее всех весов, которые видели. В конце подставляем среднее, не последнюю точку.

Последняя точка скачет вокруг минимума. Среднее этот шум съедает. Бесплатно: одна лишняя копия весов.

Сравни test с обычным `adam`.
"""
        )
    )

    cells.append(
        code(
            """fit("adam+polyak", optimizer="adam", lr=3e-3, polyak=True)
"""
        )
    )

    cells.append(
        md(
            r"""## 15. Исчезающий градиент: sigmoid vs ReLU (ch. 8.2.5)

Цепочка из пяти слоёв. Если каждый множит градиент на число $<1$, до первого слоя доезжает ноль.

Столбики — средний $|\nabla|$ по слоям после одного батча (лог-шкала).

- **sigmoid**: низ мёртвый (классика книги).
- **ReLU**: доезжает.
- **tanh + Xavier** как раз придуман, чтобы tanh *не* дох (посмотри сам, замени `"sigmoid"` → `"tanh"`).

Клип тут не поможет: крошечный градиент резать нечего.
"""
        )
    )

    cells.append(
        code(
            """if DATA is None:
    DATA = load_fashion_split(n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST, seed=SEED)
deep = (64, 64, 64, 64, 64)
rows = []
for act in ("sigmoid", "relu"):
    m = make_mlp(DATA.n_features, DATA.n_classes, hidden=deep, activation=act, init="xavier", seed=0)
    rows.append((act, one_batch_grad_profile(m, DATA.X_train, DATA.y_train)))
viz.grad_bars(rows)
"""
        )
    )

    cells.append(
        md("## 16. Табло\n\nВсё, что гонял через `fit`. Train ≫ test — переучил. Обе низкие — не дошёл / плохой шаг / мёртвый init.\n")
    )

    cells.append(code("viz.scoreboard(BOARD)\n"))

    cells.append(
        md(
            r"""## Интервью — вопрос и ответ, простым языком

Ниже те же вопросы, что в `NOTES.md`. Коротко и без лишних слов.

---

**Вопрос 1.** Почему обучение сети — это не «просто минимизируй $J$»? Что нам на самом деле важно?

**Ответ.** Нам важна ошибка на новых данных. $J$ — средняя ошибка на трейне, её мы умеем считать. Это подделка. Оптимизатор уменьшает подделку и надеется, что тест поедет следом. Если трейн падает, а тест нет — подделка соврала. Плюс мы даже не минимизируем «угадал / не угадал», а гладкую замену (кросс-энтропию).

---

**Вопрос 2.** Зачем минибатчи? Что меняется, если батч = 1 и если батч = весь трейн?

**Ответ.** Полный градиент честный, но дорогой: каждый шаг смотрит все примеры. Батч — честная оценка того же градиента, только шумная. Батч=1: дёшево, сильно трясёт, можно выпрыгнуть из острой ямы. Весь трейн: гладкий путь, мало шума, легко засесть и дорого. Обычный выбор — несколько десятков / сотен, чтобы и шум был полезный, и железо не простаивало.

---

**Вопрос 3.** Что такое плохая обусловленность? Почему SGD зигзаг? Какой самый большой безопасный шаг на квадратике?

**Ответ.** Чаша вытянутая: по одной оси ошибка крутая, по другой почти плоская. SGD одним шагом должен угодить обеим. Шаг, нормальный для плоской оси, вышибает по крутой. Получается зигзаг. На квадратике шаг должен быть меньше `2 / λ_max`, где `λ_max` — самая крутая ось. Больше — разлетается. Это слайдер §2.

---

**Вопрос 4.** Почему в глубоких сетях сёдла важнее, чем «плохие локальные минимумы»?

**Ответ.** В яме с плохим значением почти не застревают: пространства много, почти всегда есть спуск. Седло — точка, где по одним направлениям яма, по другим горка. Градиент ноль, алгоритм думает, что пришёл. Чуть шума или сдвиг в сторону горки — скатываешься. Минибатч как раз даёт этот шум. Слайдер §4: bump=0 — стоит, bump>0 — уехал.

---

**Вопрос 5.** Напиши обновления SGD и momentum. Что делает коэффициент momentum?

**Ответ.** SGD: вычти из весов шаг × градиент. Momentum: сначала обнови скорость (старая скорость × `β` минус текущий шаг), потом сдвинь веса на эту скорость. `β ≈ 0.9` значит «помни примерно последние десять шагов». Зигзаг туда-сюда гасится, движение вдоль долины разгоняется. Слишком большой `β` — проскакиваешь минимум.

---

**Вопрос 6.** Зачем Nesterov, если уже есть momentum?

**Ответ.** Обычный momentum сначала разгоняется, потом смотрит градиент *где сейчас*. Nesterov смотрит градиент *куда уже несёт* и только потом прыгает. Перед обрывом успевает притормозить. На слайдере §3 его путь обычно меньше петляет у дна.

---

**Вопрос 7.** Почему нельзя занулить все веса? Что пытается сохранить Xavier/Glorot?

**Ответ.** Если все веса нули (или все одинаковые), скрытые нейроны получают один и тот же градиент и навсегда остаются копиями. Сеть не умеет стать сложнее одного нейрона. Xavier подбирает разброс весов так, чтобы сигнал и градиент не затухали и не взрывались на первом проходе: дисперсия примерно `2 / (входы + выходы)`. Для ReLU берут He: `2 / входы`. Слайдер §12: `zero` не учится, `huge` взрывается, `xavier`/`he` едут.

---

**Вопрос 8.** AdaGrad vs RMSProp vs Adam — какую боль каждый лечит?

**Ответ.** Один глобальный шаг плох, когда координаты разной крутизны. AdaGrad делит шаг на корень из суммы квадратов прошлых градиентов: частые координаты тормозят. Сумма только растёт — к концу обучения шаг почти ноль. RMSProp то же самое, но старое забывает (скользящее среднее) — можно учить долго. Adam = RMSProp + память как у momentum + поправка, потому что счётчики стартуют с нуля. Дефолт для сетей. Не магия: большой шаг всё равно улетает.

---

**Вопрос 9.** Почему Ньютоном не учат большой нет? Что делает damping `H + λI`?

**Ответ.** Ньютон делит градиент на кривизну: на квадратике один шаг — и ты на дне. Но матрица кривизны квадратная по числу весов, её не хранить. В седле кривизна по одной оси отрицательная — Ньютон прыгает вверх. Шумная оценка кривизны с батча ещё хуже шумного градиента. `λ` на диагонали подмешивает обычный градиентный шаг: большая `λ` → почти SGD, нулевая → чистый Ньютон. Слайдер §7.

---

**Вопрос 10.** Что стабилизирует batch norm и почему после него можно поднять learning rate?

**Ответ.** По батчу приводим каждую скрытую координату к одному масштабу, потом разрешаем сети самой растянуть и сдвинуть. Чаша меньше похожа на блин, координаты сопоставимы. Шаг, который раньше вышибал одну ось, теперь нормальный. Init тоже меньше решает. На тесте берут бегущее среднее, не текущий батч — поэтому `eval()`, не `train()`.

---

**Вопрос 11.** Клип градиента — от затухания или от взрыва? Он меняет направление или только длину?

**Ответ.** От взрыва. Если длина градиента больше порога, её режут, стрелка смотрит туда же. Направление то же, шаг не гигантский. Затухание клип не лечит: умножать крошечный градиент бесполезно, его надо не дать умереть (ReLU, residual, BN, нормальный init). Слайдер §5.

---

**Вопрос 12.** Polyak averaging: что усредняют и почему среднее часто лучше последней точки?

**Ответ.** Усредняют сами веса по ходу обучения: после каждого шага чуть подмешивают текущие в бегущее среднее. Последняя точка скачет вокруг дна из-за шума батча. Среднее этот дрожащий хвост съедает. Одна лишняя копия весов, почти даром. Ячейка §14.
"""
        )
    )

    nb["cells"] = cells
    nbf.write(nb, NB)
    print("wrote", NB, "cells", len(cells))


if __name__ == "__main__":
    main()
