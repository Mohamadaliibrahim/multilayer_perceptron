import csv, argparse, random, pathlib
from typing import List

def _read_rows(path: pathlib.Path) -> List[List[str]]:
    """Return *all* non‑empty CSV rows as a list of lists."""
    with path.open(newline="") as f:
        reader = csv.reader(f)
        return [row for row in reader if row]


def _write_rows(rows: List[List[str]], path: pathlib.Path) -> None:
    """Write *rows* to *path* in CSV format (overwrites any existing file)."""
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def main() -> None:
    p = argparse.ArgumentParser(description="Split a CSV into train/valid(/test) subsets.")
    p.add_argument("--input", type=pathlib.Path, required=True, help="Original data.csv file")
    p.add_argument("--train", type=pathlib.Path, default=pathlib.Path("train.csv"))
    p.add_argument("--valid", type=pathlib.Path, default=pathlib.Path("valid.csv"))
    p.add_argument("--val_ratio",  type=float, default=0.20, help="Fraction for validation set")
    p.add_argument("--test_ratio", type=float, default=0.0,  help="Fraction for test set")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    args = p.parse_args()

    random.seed(args.seed)
    rows = _read_rows(args.input)
    random.shuffle(rows)

    n_total = len(rows)
    n_test  = int(n_total * args.test_ratio)
    n_valid = int(n_total * args.val_ratio)

    test_rows  = rows[:n_test]
    valid_rows = rows[n_test : n_test + n_valid]
    train_rows = rows[n_test + n_valid :]

    _write_rows(train_rows, args.train)
    _write_rows(valid_rows, args.valid)

    print(f"Total rows : {n_total}")
    print(f"→ train : {len(train_rows)} saved to {args.train}")
    print(f"→ valid : {len(valid_rows)} saved to {args.valid}")


if __name__ == "__main__":
    main()

#python3 split.py --input data.csv --train train.csv --valid valid.csv --seed 42