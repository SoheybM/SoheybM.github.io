"""Core implementations extracted from GPA671 Laboratory 1.

ÉTS Montréal — GPA671 — Fall 2026.
Student coursework by Soheyb Mebarbeche.

This public file focuses on the implementations used in the experiments.
Teaching statements, datasets and notebook outputs are not redistributed.
Some interface/scaffolding conventions may originate from course starter material.
"""

import numpy as np


class Perceptron:
    def __init__(self, lr: float):
        self.w = np.array([0, -1, -1], dtype=np.float64)
        self.lr = lr

    def _sigmoid(self, net: np.float64) -> np.float64:
        return 1 / (1 + np.exp(-net))

    def forward(self, x: np.ndarray) -> np.float64:
        x_aug = np.insert(x, 0, 1)
        net = self.w @ x_aug
        return self._sigmoid(net)

    def update(self, x: np.ndarray, d: np.float64) -> None:
        x_aug = np.insert(x, 0, 1)
        net = self.w @ x_aug
        y = self._sigmoid(net)
        sigmoid_prime = y * (1 - y)
        grad = (y - d) * sigmoid_prime * x_aug
        self.w = self.w - self.lr * grad


class Layer:
    def __init__(self, n_in: int, n_out: int, lr: float):
        bound = (6 / (n_in + n_out)) ** 0.5
        self.W = np.random.uniform(
            low=-bound,
            high=bound,
            size=(n_out, n_in + 1),
        )
        self.W[:, 0] = 0
        self.lr = lr
        self.last_Y = None

    def _sigmoid(self, net: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-net))

    def forward(self, X: np.ndarray) -> np.ndarray:
        X_aug = np.vstack((np.ones((1, X.shape[1])), X))
        net = self.W @ X_aug
        Y = self._sigmoid(net)
        self.last_Y = Y
        return Y

    def update(self, X: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        X_aug = np.vstack((np.ones((1, X.shape[1])), X))
        Y = self.last_Y
        sigmoid_prime = Y * (1 - Y)
        grad_net = grad_output * sigmoid_prime
        grad_input = self.W[:, 1:].T @ grad_net
        grad_W = grad_net @ X_aug.T
        self.W = self.W - self.lr * grad_W
        return grad_input


class MLP:
    def __init__(self, list_of_layers: list):
        self.list_of_layers = list_of_layers
        self.layer_inputs = []

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.layer_inputs = []
        Y = X

        for layer in self.list_of_layers:
            self.layer_inputs.append(Y)
            Y = layer.forward(Y)

        return Y

    def update(self, X: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        grad = grad_output

        for i in range(len(self.list_of_layers) - 1, -1, -1):
            layer = self.list_of_layers[i]
            layer_input = self.layer_inputs[i]
            grad = layer.update(layer_input, grad)

        return grad


def perte_regression(y_pred, y_vrai):
    return (y_pred - y_vrai) ** 2


def cout(pertes):
    return np.mean(pertes)


def perte_classification_01(y_pred, y_vrai):
    return (y_pred != y_vrai).astype(int)


def partition_holdout(X, y, proportion_val, graine=None):
    rng = np.random.default_rng(graine)
    indices = rng.permutation(len(X))
    n_val = int(round(len(X) * proportion_val))
    idx_val = indices[:n_val]
    idx_train = indices[n_val:]
    return idx_train, idx_val


def partition_k_fold(X, y, k, graine=None):
    rng = np.random.default_rng(graine)
    indices = rng.permutation(len(X))
    folds = np.array_split(indices, k)
    partitions = []

    for i in range(k):
        idx_val = folds[i]
        idx_train = np.concatenate(
            [folds[j] for j in range(k) if j != i]
        )
        partitions.append((idx_train, idx_val))

    return partitions
