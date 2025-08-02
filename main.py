from math import exp
from collections.abc import Iterable
import random

def dot_product(w: list[list[float]], x: list[float]) -> list[float]:
    
    if not w or not x or len(w[0]) != len(x):
        raise ValueError("Incompatible dimensions for W · x")

    m, n = len(w), len(x)
    result = [0.0] * m

    for i in range(m):
        total = 0.0
        for j in range(n):
            total += w[i][j] * x[j]
        result[i] = total

    return result

def add_vector(a: list[float], b: list[float]) -> list[float]:
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    return [ai + bi for ai, bi in zip(a, b)]

def sigmoid(z: float) -> float:
    """σ(z) = 1 / (1 + e^(−z))"""
    return 1.0 / (1.0 + exp(-z))

def vector_sigmoid(vec: list[float]) -> list[float]:
    return [sigmoid(v) for v in vec]


def feed_forward(w: list[list[float]], b: list[float], x: list[float]) -> tuple[list[float], list[float]]:
    z = add_vector(dot_product(w, x), b)   # z = W·x + b
    a = vector_sigmoid(z)                  # a = σ(z)
    return z, a

def forward_network(weights: list[list[list[float]]],
                    biases: list[list[float]],
                    x: list[float]) -> tuple[list[list[float]], list[float]]:
    """
    Feed-forward through an arbitrary-depth network.

    Args:
        weights : list of weight matrices, one per layer
        biases  : list of bias vectors,   one per layer
        x       : input vector

    Returns:
        zs : list of z vectors (one per layer)
        a  : final activation (network output)
    """
    if len(weights) != len(biases):
        raise ValueError("weights and biases lists must have the same length")

    activations = x
    zs = []

    for w, b in zip(weights, biases):
        z, activations = feed_forward(w, b, activations)
        zs.append(z)

    return zs, activations

if __name__ == "__main__":
    # network shape: 3 → 4 → 5 → 4 → 2
    n0, h1, h2, h3, n_out = 3, 4, 5, 4, 2

    def rand_matrix(rows, cols):
        return [[random.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]

    def rand_vector(size):
        return [random.uniform(-1, 1) for _ in range(size)]

    weights = [
        rand_matrix(h1, n0),     # W1  (4 × 3)
        rand_matrix(h2, h1),     # W2  (5 × 4)
        rand_matrix(h3, h2),     # W3  (4 × 5)
        rand_matrix(n_out, h3)   # W4  (2 × 4)
    ]

    biases = [
        rand_vector(h1),         # b1
        rand_vector(h2),         # b2
        rand_vector(h3),         # b3
        rand_vector(n_out)       # b4
    ]

    x_input = [0.6, 0.8, 1.2]    # length 3 matches n0

    zs, y_hat = forward_network(weights, biases, x_input)

    print("Layer-wise z vectors:")
    for idx, z in enumerate(zs, 1):
        print(f"  Layer {idx}: {z}")
    print("\nFinal output (a):", y_hat)