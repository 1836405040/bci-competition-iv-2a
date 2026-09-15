"""Feature extraction and leakage-safe classical decoding."""

from __future__ import annotations

import numpy as np
from mne.decoding import CSP
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class ChannelwiseScaler(BaseEstimator, TransformerMixin):
    """Fit per-channel mean/std on the training fold only."""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        self.mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.std_ = X.std(axis=(0, 2), keepdims=True) + 1e-6
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_


def make_csp_pipeline(classifier: str = "lda", n_components: int = 6, seed: int = 42) -> Pipeline:
    """Build a CSP pipeline. Every transformer is fitted inside each CV fold."""
    if classifier == "lda":
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    elif classifier == "svm":
        clf = SVC(kernel="linear", C=1.0, random_state=seed)
    else:
        raise ValueError("classifier must be 'lda' or 'svm'")
    return Pipeline([
        ("scale", ChannelwiseScaler()),
        ("csp", CSP(n_components=n_components, reg="ledoit_wolf", log=True, norm_trace=False)),
        ("classifier", clf),
    ])
