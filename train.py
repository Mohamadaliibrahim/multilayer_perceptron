import argparse, csv, pathlib, sys
import matplotlib.pyplot as plt
import numpy as np
from rules import stable_sigmoid, softmax, cross_entropy, forward, sigmoid_prime
import matplotlib.pyplot as plt

def parse_csv(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            if len(row) < 32:
                label_raw, feat_raw = row[-1], row[:-1]
            else:
                label_raw, feat_raw = row[1], row[2:]

            lbl = 1 if str(label_raw).strip().upper() in {"M", "1"} else 0
            feats.append([float(x) for x in feat_raw])
            labels.append(lbl)

    return np.asarray(feats, dtype=np.float64), np.asarray(labels, dtype=np.int64)


# ───────────────────────── network as free functions ─────────────────────────
def init_params(layer_sizes: list[int], rng: np.random.Generator):
    """He-uniform initialisation for every layer."""
    W, b = [], []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        limit = np.sqrt(6 / n_in)
        W.append(rng.uniform(-limit, limit, size=(n_in, n_out)))
        b.append(np.zeros(n_out))
    return W, b


def forward_all(X: np.ndarray, W: list[np.ndarray], b: list[np.ndarray]):
    """Return (zs, activations) for the whole net."""
    A = X
    zs, activations = [], [X]
    L = len(W)
    for l in range(L):
        Z = A @ W[l] + b[l]
        zs.append(Z)
        if l == L - 1:          # output layer → soft-max
            A = softmax(Z)
        else:                   # hidden layers → sigmoid
            A = stable_sigmoid(Z)
        activations.append(A)
    return zs, activations

def backprop_batch(X, y_hot, W, b, lr):
    """One SGD step on a mini-batch (in-place update of W, b)."""
    B = X.shape[0]
    zs, as_ = forward_all(X, W, b)

    delta = (as_[-1] - y_hot) / B                # output layer
    L = len(W)

    for l in reversed(range(L)):
        grad_W = as_[l].T @ delta
        grad_b = delta.sum(axis=0)

        W[l] -= lr * grad_W
        b[l] -= lr * grad_b

        if l != 0:
            delta = (delta @ W[l].T) * sigmoid_prime(as_[l])



# ── add just after the other imports ─────────────────────────────────────────
def plot_learning_curves(train_loss, val_loss, train_acc, val_acc, outfile: str = "learning_curves.png") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # left panel
    ax = axes[0]
    ax.plot(train_loss, label="training loss")
    ax.plot(val_loss,   label="validation loss", ls="--")
    ax.set_title("Loss")
    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")
    ax.legend()

    # right panel
    ax = axes[1]
    ax.plot(train_acc, label="training acc")
    ax.plot(val_acc,   label="validation acc", ls="--")
    ax.set_title("Accuracy")
    ax.set_xlabel("epochs")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend()

    fig.tight_layout()
    fig.savefig(outfile, dpi=120)
    print(f"> {outfile} saved")


# ─────────────────────────────── CLI / main ──────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, type=pathlib.Path)
    ap.add_argument("--valid", required=True, type=pathlib.Path)
    ap.add_argument("--layer", nargs="+", type=int, default=[64, 32, 16])
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=3.14e-2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if len(args.layer) < 2:
        ap.error("--layer needs ≥2 integers (e.g. --layer 64 32 16)")

    rng = np.random.default_rng(args.seed)

    # ─── load & scale data ────────────────────────────────────────────────
    x_train, y_train = parse_csv(args.train)
    x_valid, y_valid = parse_csv(args.valid)
    print(f"x_train shape : {x_train.shape}")
    print(f"x_valid shape : {x_valid.shape}")

    mu = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-8
    x_train = (x_train - mu) / std
    x_valid = (x_valid - mu) / std

    n_out = 2
    y_train_oh = np.eye(n_out)[y_train]
    y_valid_oh = np.eye(n_out)[y_valid]

    # ─── initialise parameters ───────────────────────────────────────────
    layer_sizes = [x_train.shape[1], *args.layer, n_out]
    W, b = init_params(layer_sizes, rng)

    pad = len(str(args.epochs))
    train_loss = []; val_loss = []
    train_acc  = []; val_acc  = []

    # ─── training loop ───────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        idx = rng.permutation(len(x_train))
        for start in range(0, len(x_train), args.batch_size):
            batch = idx[start:start + args.batch_size]
            backprop_batch(
                x_train[batch], y_train_oh[batch],
                W, b, lr=args.learning_rate
            )

        # metrics
        p_train = forward(x_train, W, b)
        p_valid = forward(x_valid, W, b)

        loss_t = cross_entropy(p_train, y_train_oh)
        loss_v = cross_entropy(p_valid, y_valid_oh)

        acc_t = (p_train.argmax(1) == y_train).mean()
        acc_v = (p_valid.argmax(1) == y_valid).mean()

        train_loss.append(loss_t); val_loss.append(loss_v)
        train_acc.append(acc_t);   val_acc.append(acc_v)

        print(f"epoch {epoch:>{pad}}/{args.epochs} - "
              f"loss: {loss_t:.4f} - val_loss: {loss_v:.4f}")

    # ─── save artefacts ──────────────────────────────────────────────────
    np.save("saved_model.npy",
        {"sizes": layer_sizes, "W": W, "b": b,
         "train_loss": train_loss, "val_loss": val_loss,
         "train_acc": train_acc, "val_acc": val_acc})
    np.savez("scaler.npz", mu=mu, std=std)
    print("> saved model (saved_model.npy) and scaler (scaler.npz)")

    plot_learning_curves(train_loss, val_loss, train_acc, val_acc)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user")

#python3 train.py --train train.csv --valid valid.csv --layer 24 24 24 --epochs 25 --batch_size 8 --learning_rate 0.0314