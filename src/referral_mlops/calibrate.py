"""Train a small model that turns extraction signals into a calibrated probability that a field is right."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from referral_mlops.evaluate import item_labels
from referral_mlops.extract.common import ALL_FIELDS
from referral_mlops.models import feature_row


def training_rows(preds, golds, texts):
    X, y = [], []
    for pred, gold, pt in zip(preds, golds, texts):
        labels = item_labels(pred, gold)
        i = 0
        for field in ALL_FIELDS:
            for r in pred.get(field, []):
                X.append(feature_row(field, r, pt))
                y.append(int(labels[i]["correct"]))
                i += 1
    return np.array(X), np.array(y)


class HeuristicCalibrator:
    """Fallback when the labelled data has no errors to learn from: keep the extractor's own score."""

    def predict_proba(self, X):
        p = np.clip(np.asarray(X)[:, 7], 0, 1)
        return np.column_stack([1 - p, p])


def fit(preds, golds, texts):
    X, y = training_rows(preds, golds, texts)
    if len(set(y)) < 2:
        return HeuristicCalibrator(), {"train_rows": int(len(y)), "train_error_rate": float(1 - y.mean()),
                                       "fallback": 1}
    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(X, y)
    p = model.predict_proba(X)[:, 1]
    stats = {"train_rows": int(len(y)), "train_error_rate": float(1 - y.mean()),
             "train_auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else 1.0,
             "train_brier": float(brier_score_loss(y, p))}
    return model, stats
