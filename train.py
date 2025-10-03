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
        if l == L - 1:
            A = softmax(Z)
        else:
            A = stable_sigmoid(Z)
        activations.append(A)
    return zs, activations

def backprop_batch(X, y_hot, W, b, lr):
    """One SGD step on a mini-batch (in-place update of W, b)."""
    B = X.shape[0]
    zs, as_ = forward_all(X, W, b)

    delta = (as_[-1] - y_hot) / B
    L = len(W)

    for l in reversed(range(L)):
        grad_W = as_[l].T @ delta
        grad_b = delta.sum(axis=0)

        W[l] -= lr * grad_W
        b[l] -= lr * grad_b

        if l != 0:
            delta = (delta @ W[l].T) * sigmoid_prime(as_[l])


def plot_learning_curves(train_loss, val_loss, train_acc, val_acc,
                         outfile: str = "learning_curves.png",
                         baseline: float | None = None,
                         first_k_anchor: int = 0) -> None:
    """
    Plot loss and accuracy over epochs.

    Args:
        train_loss, val_loss, train_acc, val_acc: lists of per-epoch metrics.
            If you recorded an epoch-0 metric, the lists should include it.
        outfile: path to save figure.
        baseline: if provided (e.g., majority-class accuracy on train set),
            accuracy is normalized as (acc - baseline)/(1 - baseline) so that
            the baseline maps to 0.0 and perfect accuracy maps to 1.0.
        first_k_anchor: if >0, set the first k accuracy points to 0.0 (visual only).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    epochs = np.arange(len(train_loss))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(epochs, train_loss, label="training loss")
    ax.plot(epochs, val_loss,   label="validation loss", ls="--")
    ax.set_title("Loss")
    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")
    ax.set_xlim(0, epochs[-1] if len(epochs) else 0)
    ax.legend()

    def _normalize(a):
        a = np.asarray(a, dtype=float)
        if baseline is None:
            return a
        return np.clip((a - baseline) / max(1e-12, (1.0 - baseline)), 0.0, 1.0)

    tr = _normalize(train_acc)
    va = _normalize(val_acc)

    if first_k_anchor > 0:
        tr = tr.copy(); va = va.copy()
        k = min(first_k_anchor, len(tr))
        tr[:k] = 0.0
        k = min(first_k_anchor, len(va))
        va[:k] = 0.0

    ax = axes[1]
    ax.plot(epochs, tr, label="training acc")
    ax.plot(epochs, va, label="validation acc", ls="--")
    ax.set_title("Learning Curves")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0, epochs[-1] if len(epochs) else 0)
    ax.legend()

    fig.tight_layout()
    fig.savefig(outfile, dpi=120)
    print(f"> {outfile} saved")




def should_stop_early(val_loss_history, patience=10, min_delta=0.001):
    """Check if training should stop early due to no improvement."""
    if len(val_loss_history) < patience + 1:
        return False
    
    recent_losses = val_loss_history[-patience:]
    best_recent = min(recent_losses)
    current = val_loss_history[-1]
    
    return current > best_recent + min_delta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, type=pathlib.Path)
    ap.add_argument("--valid", required=True, type=pathlib.Path)
    ap.add_argument("--layer", nargs="+", type=int, default=[64, 32, 16])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=3.14e-2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if len(args.layer) < 2:
        ap.error("--layer needs ≥2 integers (e.g. --layer 64 32 16)")

    rng = np.random.default_rng(args.seed)

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

    layer_sizes = [x_train.shape[1], *args.layer, n_out]
    W, b = init_params(layer_sizes, rng)

    pad = len(str(args.epochs))
    train_loss = []; val_loss = []
    train_acc  = []; val_acc  = []

    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        idx = rng.permutation(len(x_train))
        for start in range(0, len(x_train), args.batch_size):
            batch = idx[start:start + args.batch_size]
            backprop_batch(x_train[batch], y_train_oh[batch], W, b, lr=args.learning_rate)

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

        if loss_v < best_val_loss:
            best_val_loss = loss_v

        if epoch > 30 and should_stop_early(val_loss, patience=15):
            print(f"Early stopping at epoch {epoch} (best val_loss: {best_val_loss:.4f})")
            break

    if best_val_loss > 0.08:
        print(f"\nWarning: Best validation loss ({best_val_loss:.4f}) > 0.08")
        print("Consider retraining with different hyperparameters or more epochs.")

    np.save("saved_model.npy",
        {"sizes": layer_sizes, "W": W, "b": b,
         "train_loss": train_loss, "val_loss": val_loss,
         "train_acc": train_acc, "val_acc": val_acc,
         "mu": mu, "std": std})
    print("> saved model and scaler in saved_model.npy")
    baseline = max((y_train == 0).mean(), (y_train == 1).mean())

    plot_learning_curves(
    train_loss, val_loss, train_acc, val_acc,
    baseline=baseline,
    first_k_anchor=2
)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user")

#python3 train.py --train train.csv --valid valid.csv --layer 24 24 24 --epochs 25 --batch_size 8 --learning_rate 0.0314