#!/usr/bin/env python3
"""
predict.py – load a saved MLP (+ scaler) and evaluate / predict a CSV file.
usage:
    python3 predict.py --model saved_model.npy --test some.csv
"""

import argparse, csv, pathlib, sys
import numpy as np

# ─────────────────────── CSV loader ────────────────────────────────
def read_csv(path):
    feats, labels = [], []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            label_raw = row[1] if len(row) >= 32 else row[-1]
            feat_raw  = row[2:] if len(row) >= 32 else row[:-1]
            labels.append(1 if str(label_raw).strip().upper() in {"M","1"} else 0)
            feats.append([float(x) for x in feat_raw])
    return np.asarray(feats, np.float64), np.asarray(labels, np.int64)

# ─────────────────────── activations ───────────────────────────────
def stable_sigmoid(z):
    pos = z >= 0
    out = np.empty_like(z)
    out[pos]  = 1.0 / (1.0 + np.exp(-z[pos]))
    ez        = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)

def forward(X, W, b):
    A = X
    for i in range(len(W)):
        Z = A @ W[i] + b[i]
        A = softmax(Z) if i == len(W)-1 else stable_sigmoid(Z)
    return A

def cross_entropy(pred, y_hot):
    eps = 1e-12
    pred = np.clip(pred, eps, 1-eps)
    return -np.mean(np.sum(y_hot*np.log(pred), axis=1))

# ─────────────────────────── main ──────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=pathlib.Path)
    ap.add_argument("--test",  required=True, type=pathlib.Path)
    args = ap.parse_args()

    artefact = np.load(args.model, allow_pickle=True).item()

    # safe key handling ------------------------------------------------------
    if "W" in artefact and "b" in artefact:            # newest train.py
        W, b = artefact["W"], artefact["b"]
    elif "weights" in artefact and "biases" in artefact:  # older format
        W, b = artefact["weights"], artefact["biases"]
    else:
        raise KeyError("Cannot find weight / bias matrices in saved model")

    scaler = np.load("scaler.npz")
    mu, std = scaler["mu"], scaler["std"]

    # load test set ----------------------------------------------------------
    X, y = read_csv(args.test)
    X = (X - mu) / std

    prob = forward(X, W, b)
    pred = prob.argmax(1)

    acc  = (pred == y).mean()*100
    loss = cross_entropy(prob, np.eye(2)[y])

    print(f"Test accuracy        : {acc:.2f}%  ({len(y)} samples)")
    print(f"Binary cross-entropy : {loss:.4f}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted")
