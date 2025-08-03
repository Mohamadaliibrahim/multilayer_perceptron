#!/usr/bin/env python3
# main.py  –  3-hidden-layer MLP for the WDBC breast-cancer dataset
from math import exp, sqrt
import random, csv, math, pathlib


# ───────────────────── low-level helpers ─────────────────────────────────────
def dot_product(w, x):
    if not w or not x or len(w[0]) != len(x):
        raise ValueError("Incompatible dimensions for W · x")
    return [sum(wi * xi for wi, xi in zip(w_row, x)) for w_row in w]


def add_vector(a, b):
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    return [ai + bi for ai, bi in zip(a, b)]


# numerically-stable logistic sigmoid
def sigmoid(z: float) -> float:
    if z >= 0:
        ez = exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = exp(z)
        return ez / (1.0 + ez)


def vector_sigmoid(vec):
    return [sigmoid(v) for v in vec]


def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1.0 - s)


# numerically-stable soft-max
def vector_softmax(vec):
    m = max(vec)
    exps = [exp(v - m) for v in vec]
    s = sum(exps)
    return [e / s for e in exps]


def outer(col, row):
    return [[c * r for r in row] for c in col]


# ───────────────────── forward / backward passes ─────────────────────────────
def forward(weights, biases, x):
    zs, activs = [], [x]
    for k,(w,b) in enumerate(zip(weights,biases), 1):
        z = add_vector(dot_product(w, activs[-1]), b)
        a = vector_softmax(z) if k == len(weights) else vector_sigmoid(z)
        zs.append(z); activs.append(a)
    return zs, activs

def backprop(weights, biases, x, y, lr):
    zs, as_ = forward(weights, biases, x)
    delta = [as_[-1][k] - y[k] for k in range(len(y))]          # output layer

    for l in reversed(range(len(weights))):
        grad_w = outer(delta, as_[l])
        for i in range(len(weights[l])):
            for j in range(len(weights[l][0])):
                weights[l][i][j] -= lr * grad_w[i][j]
            biases[l][i] -= lr * delta[i]

        if l:                               # back-prop to previous layer
            wt = list(zip(*weights[l]))     # Wᵗ
            delta = [sum(wt_i[k]*delta[k] for k in range(len(delta)))
                     * sigmoid_prime(zs[l-1][i])
                     for i, wt_i in enumerate(wt)]


# ───────────────────── random initialisers ───────────────────────────────────
def glorot(rows, cols):
    limit = sqrt(6/(rows+cols))
    return [[random.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]

def rand_vec(n): return [random.uniform(-0.1,0.1) for _ in range(n)]


# ───────────────────── CSV loader ────────────────────────────────────────────
def load_wdbc(path, expect=30):
    ids,lbl,feat=[],[],[]
    with open(path,newline='') as f:
        for ln,row in enumerate(csv.reader(f),1):
            if not row: continue
            if len(row)!=2+expect: raise ValueError(f"Line {ln}: wrong cols")
            ids.append(row[0]); lbl.append(1 if row[1].strip().upper()=="M" else 0)
            feat.append([float(x) for x in row[2:]])
    return ids,lbl,feat


# ───────────────────── main ──────────────────────────────────────────────────
if __name__ == "__main__":
    CSV_PATH   = pathlib.Path("data.csv")
    HIDDEN     = [64,32,16]
    LR         = 0.02
    EPOCHS     = 20
    TEST_SPLIT = 0.20
    random.seed(42)

    ids,lbl,feat = load_wdbc(CSV_PATH)

    # z-score standardise
    cols = list(zip(*feat))
    means = [sum(c)/len(c) for c in cols]
    stds  = [math.sqrt(sum((x-m)**2 for x in c)/len(c)) for c,m in zip(cols,means)]
    feat  = [[(x-m)/(s or 1) for x,m,s in zip(r,means,stds)] for r in feat]

    # shuffle & split
    data=list(zip(ids,feat,lbl)); random.shuffle(data)
    split = int(len(data)*(1-TEST_SPLIT))
    train,test = data[:split],data[split:]
    Xtr,Ytr = [f for _,f,_ in train],[y for *_,y in train]
    Xte,Yte = [f for _,f,_ in test ],[y for *_,y in test ]

    # net shapes
    sizes=[len(Xtr[0])]+HIDDEN+[2]      # 2-class soft-max
    W=[glorot(o,i) for i,o in zip(sizes[:-1],sizes[1:])]
    b=[rand_vec(s) for s in sizes[1:]]

    # one-hot with *matching indices*  (label 0→[1,0], label 1→[0,1])
    oh = lambda y: [1,0] if y==0 else [0,1]

    # training
    for epoch in range(1,EPOCHS+1):
        for x,y in random.sample(list(zip(Xtr,Ytr)), len(Xtr)):
            backprop(W,b,x,oh(y),LR)

        preds=[forward(W,b,x)[1][-1].index(max(forward(W,b,x)[1][-1])) for x in Xtr]
        acc=sum(p==y for p,y in zip(preds,Ytr))/len(Ytr)
        print(f"Epoch {epoch:3d}/{EPOCHS} — train acc: {acc:.3%}")

    # test
    preds=[forward(W,b,x)[1][-1].index(max(forward(W,b,x)[1][-1])) for x in Xte]
    acc=sum(p==y for p,y in zip(preds,Yte))/len(Yte)
    print(f"\nTest accuracy: {acc:.3%} on {len(Yte)} samples")