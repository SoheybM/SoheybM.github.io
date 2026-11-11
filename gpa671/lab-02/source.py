"""Core implementations extracted from GPA671 Laboratory 2.

ÉTS Montréal — GPA671 — Fall 2026.
Student coursework by Soheyb Mebarbeche.

This public file focuses on custom implementations used in the experiments.
Teaching statements, datasets and course-provided starter code are not redistributed.
"""

import numpy as np


def standardiser(X_ref, *autres):
    """Standardize datasets using statistics computed only from X_ref."""
    moyenne = np.mean(X_ref, axis=0)
    ecart_type = np.std(X_ref, axis=0)
    resultats = [(X_ref - moyenne) / ecart_type]

    for X_autre in autres:
        resultats.append((X_autre - moyenne) / ecart_type)

    return resultats


def kfold_indices(n, k, seed):
    """Return k (training_indices, validation_indices) pairs."""
    rng = np.random.default_rng(seed)
    groupes = np.array_split(rng.permutation(n), k)

    return [
        (
            np.concatenate([groupes[j] for j in range(k) if j != i]),
            groupes[i],
        )
        for i in range(k)
    ]


def rbf_kernel_matrix(A, B, gamma):
    """Compute an RBF kernel matrix without a Python loop."""
    norme_A = np.sum(A ** 2, axis=1)[:, np.newaxis]
    norme_B = np.sum(B ** 2, axis=1)[np.newaxis, :]
    distances_carre = norme_A + norme_B - 2 * A @ B.T
    distances_carre = np.maximum(distances_carre, 0)

    return np.exp(-gamma * distances_carre)


def fonction_decision_duale(model, X, gamma):
    """Compute the SVM decision function from its dual representation."""
    K = rbf_kernel_matrix(X, model.support_vectors_, gamma)
    return K @ model.dual_coef_[0] + model.intercept_[0]


class kNN:
    def __init__(self, k: int, weights: str = "uniform", eps: float = 1e-12):
        if k <= 0:
            raise ValueError("k must be strictly positive.")
        if weights not in ("uniform", "distance"):
            raise ValueError("weights must be 'uniform' or 'distance'.")

        self.k = k
        self.weights = weights
        self.eps = eps
        self.X_train = None
        self.y_train = None
        self.classes_ = None
        self.y_indices = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X)
        y = np.asarray(y)

        if self.k > len(X):
            raise ValueError("k cannot exceed the number of training examples.")

        self.X_train = X
        self.y_train = y
        self.classes_ = np.unique(y)
        self.y_indices = np.searchsorted(self.classes_, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None:
            raise RuntimeError("The model must be fitted before prediction.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        distances_carre = (
            np.sum(X ** 2, axis=1)[:, None]
            + np.sum(self.X_train ** 2, axis=1)[None, :]
            - 2 * X @ self.X_train.T
        )
        distances = np.sqrt(np.maximum(distances_carre, 0))

        voisins = np.argsort(
            distances,
            axis=1,
            kind="stable",
        )[:, : self.k]

        distances_voisins = np.take_along_axis(
            distances,
            voisins,
            axis=1,
        )
        classes_voisins = self.y_indices[voisins]

        if self.weights == "uniform":
            poids = np.ones_like(distances_voisins)
        else:
            poids = 1.0 / (distances_voisins + self.eps)

        one_hot = np.eye(len(self.classes_))[classes_voisins]
        scores = np.sum(one_hot * poids[:, :, None], axis=1)

        return scores / np.sum(poids, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


class NaiveBayes:
    def __init__(self, var_smoothing: float = 1e-09):
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.class_log_prior_ = None
        self.theta_ = None
        self.var_ = None
        self.epsilon_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_, counts = np.unique(y, return_counts=True)
        self.class_log_prior_ = np.log(counts / len(y))

        self.theta_ = np.array(
            [np.mean(X[y == classe], axis=0) for classe in self.classes_]
        )
        self.var_ = np.array(
            [
                np.var(X[y == classe], axis=0, ddof=0)
                for classe in self.classes_
            ]
        )

        self.epsilon_ = (
            self.var_smoothing
            * np.max(np.var(X, axis=0, ddof=0))
        )
        self.var_ += self.epsilon_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        diff = X[:, None, :] - self.theta_[None, :, :]

        log_likelihood = -0.5 * np.sum(
            np.log(2 * np.pi * self.var_[None, :, :])
            + diff ** 2 / self.var_[None, :, :],
            axis=2,
        )
        log_joint = log_likelihood + self.class_log_prior_[None, :]

        maximum = np.max(log_joint, axis=1, keepdims=True)
        log_norm = maximum + np.log(
            np.sum(np.exp(log_joint - maximum), axis=1, keepdims=True)
        )

        return np.exp(log_joint - log_norm)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
