#!/usr/bin/env python3
"""
train.py

• loads a train CSV and a validation CSV
• builds an MLP with ≥2 hidden layers (sizes given on the CLI)
• standardises every feature (μ/σ computed from the *training* set)
• trains with mini-batch SGD + back-propagation + cross-entropy + softmax
• prints one progress line per epoch (train / val loss)
• saves
      - the network (topology + weights + biases)   → saved_model.npy
      - the scaler (μ, σ)                           → scaler.npz
      - two learning-curve graphs (loss & accuracy) → learning_curves.png
Usage example
-------------
python3 train.py \
        --train train.csv --valid valid.csv \
        --layer 64 32 16         \
        --epochs 70              \
        --batch_size 8           \
        --learning_rate 0.0314
"""

import argparse
import csv
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

# ────────────────────────────── helpers ──────────────────────────────────────
def stable_sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically-stable logistic sigmoid (vectorised)."""
    pos = z >= 0
    neg = ~pos
    out = np.empty_like(z)
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def sigmoid_prime(a: np.ndarray) -> np.ndarray:
    """Derivative w.r.t. the *activation* (already σ(z))."""
    return a * (1.0 - a)


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise soft-max (numerically safe)."""
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def cross_entropy(pred: np.ndarray, target_hot: np.ndarray) -> float:
    """Mean cross-entropy for one-hot targets."""
    eps = 1e-12
    pred = np.clip(pred, eps, 1.0 - eps)
    return -np.mean(np.sum(target_hot * np.log(pred), axis=1))


# ─────────────────────────── data handling ───────────────────────────────────
def parse_csv(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads a Breast-Cancer WDBC-style CSV.

    Expected layout (len(row) == 32):
       0 : sample id (ignored)
       1 : label  'M' / 'B'   OR 1 / 0
     2-31 : 30 float features
    """
    feats, labels = [], []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            if len(row) < 32:  # allow train/valid CSV that dropped the id
                # assume 31 columns: 30 features + label at end
                label_raw = row[-1]
                feat_raw = row[:-1]
            else:
                label_raw = row[1]
                feat_raw = row[2:]

            # label → 0/1
            if str(label_raw).strip().upper() in {"M", "1"}:
                lbl = 1
            else:
                lbl = 0

            feats.append([float(x) for x in feat_raw])
            labels.append(lbl)

    return np.asarray(feats, dtype=np.float64), np.asarray(labels, dtype=np.int64)


# ───────────────────────────── network class ─────────────────────────────────
class MLP:
    """A very small Numpy-only MLP with soft-max output."""

    def __init__(self, layer_sizes: list[int], rng: np.random.Generator):
        """
        layer_sizes: e.g. [30, 64, 32, 16, 2]
                      input ─┘  ↑   ↑   ↑   └─ output
        """
        self.sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1  # without input layer
        self.W = []  # weights: list of (in, out)
        self.b = []  # biases : list of (out, )

        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # He-uniform initialisation (good for sigmoid)
            limit = np.sqrt(6 / n_in)
            self.W.append(rng.uniform(-limit, limit, size=(n_in, n_out)))
            self.b.append(np.zeros(n_out))

    # ────────────── forward & helpers ────────────────────────
    def _forward_all(self, X: np.ndarray):
        """
        Returns:
        zs: list of pre-activations
        as: list of activations (a0 … aL)
        """
        A = X
        zs, activations = [], [X]

        for i in range(self.n_layers):
            Z = A @ self.W[i] + self.b[i]
            zs.append(Z)
            if i == self.n_layers - 1:  # last → softmax
                A = softmax(Z)
            else:
                A = stable_sigmoid(Z)
            activations.append(A)

        return zs, activations

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Only the final activations (probabilities)."""
        _, acts = self._forward_all(X)
        return acts[-1]

    # ───────────── back-prop for one mini-batch ──────────────
    def backprop(self, X: np.ndarray, y_hot: np.ndarray, lr: float):
        """
        One gradient-descent step for a whole mini-batch.
        Shapes:
          X      : (B, n_in)
          y_hot  : (B, n_out)
        """
        B = X.shape[0]
        zs, as_ = self._forward_all(X)

        # output layer delta
        delta = (as_[-1] - y_hot) / B  # (B, n_out)

        for l in reversed(range(self.n_layers)):
            # grads (in, out)
            grad_W = as_[l].T @ delta
            grad_b = delta.sum(axis=0)

            # update
            self.W[l] -= lr * grad_W
            self.b[l] -= lr * grad_b

            if l != 0:
                # propagate delta
                delta = (delta @ self.W[l].T) * sigmoid_prime(as_[l])


# ─────────────────────────────── CLI / main ──────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, type=pathlib.Path, help="train CSV")
    ap.add_argument("--valid", required=True, type=pathlib.Path, help="valid CSV")
    ap.add_argument("--layer", nargs="+", type=int, default=[64, 32, 16],
                    help="hidden layer sizes, e.g. --layer 64 32 16")
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=3.14e-2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # ─── load data ──────────────────────────────────────────────────────────
    x_train, y_train = parse_csv(args.train)
    x_valid, y_valid = parse_csv(args.valid)

    print(f"x_train shape : {x_train.shape}")
    print(f"x_valid shape : {x_valid.shape}")

    # ─── feature scaling (μ/σ) ──────────────────────────────────────────────
    mu = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-8
    x_train = (x_train - mu) / std
    x_valid = (x_valid - mu) / std

    # ─── one-hot encode labels ──────────────────────────────────────────────
    n_out = 2
    y_train_oh = np.eye(n_out)[y_train]
    y_valid_oh = np.eye(n_out)[y_valid]

    # ─── build network ──────────────────────────────────────────────────────
    layer_sizes = [x_train.shape[1], *args.layer, n_out]
    net = MLP(layer_sizes, rng)

    # for reproducible printing width
    pad = len(str(args.epochs))

    # containers for learning curves
    train_loss, val_loss = [], []
    train_acc,  val_acc  = [], []

    # ─── training loop ──────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        # mini-batch shuffle
        idx = rng.permutation(len(x_train))
        for start in range(0, len(x_train), args.batch_size):
            batch_idx = idx[start: start + args.batch_size]
            net.backprop(
                x_train[batch_idx],
                y_train_oh[batch_idx],
                lr=args.learning_rate
            )

        # metrics after epoch
        p_train = net.forward(x_train)
        p_valid = net.forward(x_valid)

        loss_t = cross_entropy(p_train, y_train_oh)
        loss_v = cross_entropy(p_valid, y_valid_oh)

        acc_t = (p_train.argmax(1) == y_train).mean()
        acc_v = (p_valid.argmax(1) == y_valid).mean()

        train_loss.append(loss_t)
        val_loss.append(loss_v)
        train_acc.append(acc_t)
        val_acc.append(acc_v)

        print(f"epoch {epoch:>{pad}}/{args.epochs} - "
              f"loss: {loss_t:.4f} - val_loss: {loss_v:.4f}")

    # ─── save artefacts ────────────────────────────────────────────────────
    np.save("saved_model.npy",
            {"sizes": net.sizes, "W": net.W, "b": net.b})
    np.savez("scaler.npz", mu=mu, std=std)
    print("> saved model (saved_model.npy) and scaler (scaler.npz)")

    # ─── learning-curve plots ──────────────────────────────────────────────
    plt.figure(figsize=(11, 4))

    # left: loss
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="training loss")
    plt.plot(val_loss, label="validation loss", linestyle="--")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("Loss")
    plt.legend()

    # right: accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="training acc")
    plt.plot(val_acc, label="validation acc", linestyle="--")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1)
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=120)
    print("> learning_curves.png saved")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user")
