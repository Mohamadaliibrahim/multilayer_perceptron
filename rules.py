import numpy as np

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
    total_loss = 0.0
    N = len(pred)

    for i in range(N):
        p_M, p_B = pred[i]
        y_M, y_B = y_hot[i]

        p_M = max(min(p_M, 1-eps), eps)
        p_B = max(min(p_B, 1-eps), eps)

        sample_loss = - (y_M * np.log(p_M) + y_B * np.log(p_B))
        total_loss += sample_loss

    return total_loss / N

def sigmoid_prime(a: np.ndarray) -> np.ndarray:
    """Derivative w.r.t. the *activation* (already σ(z))."""
    return a * (1.0 - a)
