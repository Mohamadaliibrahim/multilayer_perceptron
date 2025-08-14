import argparse, csv, pathlib, sys
import numpy as np
from rules import cross_entropy, forward

def read_csv(path: pathlib.Path):
    ids, feats, labels = [], [], []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue

            if len(row) >= 32:
                sample_id, label_raw, feat_raw = row[0], row[1], row[2:]
            else:
                sample_id = f"idx_{len(ids)}"
                label_raw, feat_raw = row[-1], row[:-1]

            lbl = 1 if str(label_raw).strip().upper() in {"M", "1"} else 0
            feats.append([float(x) for x in feat_raw])
            labels.append(lbl)
            ids.append(sample_id)

    feats = np.asarray(feats, np.float64)
    labels = np.asarray(labels, np.int64)
    ids = np.asarray(ids, dtype=object)
    return ids, feats, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=pathlib.Path,
                    help="saved_model.npy produced by train.py")
    ap.add_argument("--test",  required=True, type=pathlib.Path,
                    help="CSV to evaluate / predict")
    args = ap.parse_args()

    artefact = np.load(args.model, allow_pickle=True).item()

    if "W" in artefact and "b" in artefact:
        W, b = artefact["W"], artefact["b"]
    elif "weights" in artefact and "biases" in artefact:
        W, b = artefact["weights"], artefact["biases"]
    else:
        raise KeyError(" Could not find weight/bias matrices in saved model")

    # Load normalization parameters from the same file
    if "mu" in artefact and "std" in artefact:
        mu, std = artefact["mu"], artefact["std"]
    else:
        # Fallback to old scaler.npz for backward compatibility
        try:
            scaler = np.load("scaler.npz")
            mu, std = scaler["mu"], scaler["std"]
        except FileNotFoundError:
            sys.exit(" Normalization parameters not found in model file or scaler.npz")

    ids, X, y = read_csv(args.test)
    X = (X - mu) / std

    prob = forward(X, W, b)
    pred = prob.argmax(1)

    acc  = (pred == y).mean() * 100.0
    loss = cross_entropy(prob, np.eye(2)[y])

    print(f"Test accuracy        : {acc:.2f}%  ({len(y)} samples)")
    print(f"Binary cross-entropy : {loss:.4f}")

    print("\n#  sample_id         true  pred   p(M)    p(B)")
    for i, (sid, t, p, pr) in enumerate(zip(ids, y, pred, prob), start=1):
        print(f"{i:3d}  {sid:>14}    {t}     {p}   {pr[1]:.4f}  {pr[0]:.4f}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user")


#python3 predict.py --model saved_model.npy --test valid.csv