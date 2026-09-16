"""Cross-validation routines and metrics."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline


@dataclass
class CVResult:
    subject: int
    model: str
    accuracy_mean: float
    accuracy_std: float
    balanced_accuracy_mean: float
    n_trials: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def evaluate_classical(X: np.ndarray, y: np.ndarray, pipeline: Pipeline, subject: int,
                       model_name: str, folds: int = 5, seed: int = 42) -> CVResult:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=["accuracy", "balanced_accuracy"], n_jobs=1)
    return CVResult(subject, model_name, float(scores["test_accuracy"].mean()),
                    float(scores["test_accuracy"].std()),
                    float(scores["test_balanced_accuracy"].mean()), int(len(y)))


def evaluate_eegnet(X: np.ndarray, y: np.ndarray, subject: int, *, folds: int = 5,
                    seed: int = 42, epochs: int = 100) -> CVResult:
    """Evaluate EEGNet with fold-local training and validation."""
    from .models import fit_eegnet

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    accuracies: list[float] = []
    balanced_accuracies: list[float] = []
    for train_idx, valid_idx in cv.split(X, y):
        _, history = fit_eegnet(X[train_idx], y[train_idx], X[valid_idx], y[valid_idx],
                                epochs=epochs, seed=seed)
        accuracies.append(history["valid_accuracy"][-1])
        balanced_accuracies.append(history["valid_balanced_accuracy"][-1])
    return CVResult(subject, "eegnet", float(np.mean(accuracies)), float(np.std(accuracies)),
                    float(np.mean(balanced_accuracies)), int(len(y)))
