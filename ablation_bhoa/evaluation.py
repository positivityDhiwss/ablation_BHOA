"""
evaluation.py
=============
Fungsi fitness wrapper (KNN, balanced accuracy) dan evaluasi akhir pada data uji terkunci.

    F(b) = 0.99 * [1 - BA_validation(b)] + 0.01 * (|b| / d)      (diminimalkan)
"""
from __future__ import annotations

import time
from typing import Callable, Dict

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier

from .data import FinalData, SearchData


def make_knn(k: int = 5, n_jobs: int = 1) -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=k, metric="euclidean", weights="uniform",
                                algorithm="brute", n_jobs=n_jobs)


def make_fitness(sd: SearchData, alpha: float = 0.99, k: int = 5) -> Callable[[np.ndarray], float]:
    d = len(sd.prep.groups)

    def fitness(mask: np.ndarray) -> float:
        cols = sd.prep.columns_for(mask)
        if cols.size == 0:
            return 1.0
        clf = make_knn(k).fit(sd.X_tr[:, cols], sd.y_tr)
        ba = balanced_accuracy_score(sd.y_val, clf.predict(sd.X_val[:, cols]))
        return alpha * (1.0 - ba) + (1.0 - alpha) * (mask.sum() / d)

    return fitness


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Kelas serangan (1) = kelas positif. Pembagi nol -> skor 0 (zero_division=0)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_attack": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
        "precision_attack": prec,
        "recall_attack": rec,
        "balanced_accuracy": 0.5 * (rec + tnr),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


class FinalEvaluator:
    """Melatih ulang KNN pada data latih akhir dengan mask terpilih lalu menguji pada data uji."""

    def __init__(self, fd: FinalData, k: int = 5, n_jobs: int = -1):
        self.fd, self.k, self.n_jobs = fd, k, n_jobs
        self._cache: Dict[bytes, Dict[str, float]] = {}

    def evaluate(self, mask: np.ndarray) -> Dict[str, float]:
        key = np.packbits(mask.astype(np.uint8)).tobytes()
        if key in self._cache:
            return dict(self._cache[key])
        cols = self.fd.prep.columns_for(mask)
        clf = make_knn(self.k, self.n_jobs)
        t0 = time.perf_counter()
        clf.fit(self.fd.X_tr[:, cols], self.fd.y_tr)
        t1 = time.perf_counter()
        y_pred = clf.predict(self.fd.X_te[:, cols])
        t2 = time.perf_counter()
        out = classification_metrics(self.fd.y_te, y_pred)
        out["final_fit_s"] = t1 - t0
        out["final_predict_s"] = t2 - t1
        self._cache[key] = out
        return dict(out)
