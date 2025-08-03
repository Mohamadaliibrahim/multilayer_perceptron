#!/usr/bin/env python3
"""A small 3‑hidden‑layer MLP for the Breast‑Cancer WDBC data.
   Progress lines now follow the Keras‑style format:
   epoch N/70 - loss: x.xxxx - val_loss: y.yyyy
"""
from __future__ import annotations
from math import exp, sqrt, log
import csv, math, pathlib, random, sys

# ───────────────────── vector helpers ────────────────────────────────────────

def dot(w, x):
    """Matrix–vector dot product (row‑major)."""
    return [sum(wij * xj for wij, xj in zip(w_row, x)) for w_row in w]

def add(a, b):
    return [ai + bi for ai, bi in zip(a, b)]

def sigmoid(z: float) -> float:
    if z >= 0:
        ez = exp(-z)
        return 1.0 / (1.0 + ez)
    ez = exp(z)
    return ez / (1.0 + ez)

def vec_sigmoid(v):
    return [sigmoid(z) for z in v]

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1.0 - s)

def softmax(v):
    m = max(v)
    exps = [exp(z - m) for z in v]
    s = sum(exps)
    return [e / s for e in exps]

def outer(col, row):
    return [[c * r for r in row] for c in col]

# ───────────────────── forward / back‑prop ───────────────────────────────────

def forward(W, b, x):
    zs, activs = [], [x]
    for k, (w, bi) in enumerate(zip(W, b), 1):
        z = add(dot(w, activs[-1]), bi)
        a = softmax(z) if k == len(W) else vec_sigmoid(z)
        zs.append(z)
        activs.append(a)
    return zs, activs

def backprop(W, b, x, y, lr):
    zs, a = forward(W, b, x)
    delta = [a[-1][k] - y[k] for k in range(len(y))]  # output layer
    for l in reversed(range(len(W))):
        grad_w = outer(delta, a[l])
        # update
        for i in range(len(W[l])):
            for j in range(len(W[l][0])):
                W[l][i][j] -= lr * grad_w[i][j]
            b[l][i] -= lr * delta[i]
        if l:
            wt = list(zip(*W[l]))  # transpose
            delta = [sum(wt_i[k] * delta[k] for k in range(len(delta))) * sigmoid_prime(zs[l - 1][i])
                     for i, wt_i in enumerate(wt)]

# ───────────────────── utilities ─────────────────────────────────────────────

def glorot(rows: int, cols: int):
    limit = sqrt(6 / (rows + cols))
    return [[random.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]

def rand_vec(n):
    return [random.uniform(-0.1, 0.1) for _ in range(n)]

def cross_entropy(pred, target, eps: float = 1e-15):
    return -sum(t * log(max(p, eps)) for p, t in zip(pred, target))

# ───────────────────── data loader ───────────────────────────────────────────

def load_wdbc(path: pathlib.Path, expect: int = 30):
    ids, lbl, feat = [], [], []
    with open(path, newline="") as f:
        for ln, row in enumerate(csv.reader(f), 1):
            if not row:
                continue
            if len(row) != 2 + expect:
                raise ValueError(f"Line {ln}: expected {2 + expect} cols, got {len(row)}")
            ids.append(row[0])
            lbl.append(1 if row[1].strip().upper() == "M" else 0)
            feat.append([float(x) for x in row[2:]])
    return ids, lbl, feat

# ───────────────────── main ─────────────────────────────────────────────────

def main(argv: list[str] | None = None):
    CSV_PATH = pathlib.Path("data.csv")
    HIDDEN = [64, 32, 16]
    LR = 0.02
    EPOCHS = 20
    TEST_SPLIT = 0.2
    random.seed(42)

    ids, y, X = load_wdbc(CSV_PATH)

    # z‑score standardisation
    cols = list(zip(*X))
    means = [sum(c) / len(c) for c in cols]
    stds = [math.sqrt(sum((v - m) ** 2 for v in c) / len(c)) for c, m in zip(cols, means)]
    X = [[(v - m) / (s or 1) for v, m, s in zip(r, means, stds)] for r in X]

    data = list(zip(X, y))
    random.shuffle(data)
    split = int(len(data) * (1 - TEST_SPLIT))
    train, val = data[:split], data[split:]
    Xtr, Ytr = [x for x, _ in train], [y for _, y in train]
    Xval, Yval = [x for x, _ in val], [y for _, y in val]

    sizes = [len(Xtr[0])] + HIDDEN + [2]
    W = [glorot(o, i) for i, o in zip(sizes[:-1], sizes[1:])]
    b = [rand_vec(s) for s in sizes[1:]]

    # one‑hot where label 0→[1,0], label 1→[0,1]
    one_hot = lambda lbl: [1, 0] if lbl == 0 else [0, 1]

    # training loop ---------------------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        for x_vec, lbl in random.sample(list(zip(Xtr, Ytr)), len(Xtr)):
            backprop(W, b, x_vec, one_hot(lbl), LR)

        # ------- metrics ----------------------------------------------------
        tr_loss = sum(cross_entropy(forward(W, b, x)[1][-1], one_hot(y)) for x, y in zip(Xtr, Ytr)) / len(Xtr)
        val_loss = sum(cross_entropy(forward(W, b, x)[1][-1], one_hot(y)) for x, y in zip(Xval, Yval)) / len(Xval)
        print(f"epoch {epoch:2d}/{EPOCHS} - loss: {tr_loss:.4f} - val_loss: {val_loss:.4f}")

    # final evaluation ------------------------------------------------------
    preds = [forward(W, b, x)[1][-1].index(max(forward(W, b, x)[1][-1])) for x in Xval]
    acc = sum(p == y for p, y in zip(preds, Yval)) / len(Yval)
    print(f"\nValidation accuracy: {acc:.3%} on {len(Yval)} samples")

if __name__ == "__main__":
    sys.exit(main())
