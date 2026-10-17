"""Population stability index on document features and model confidence."""
from __future__ import annotations

import numpy as np

DRIFT_FEATURES = ["ocr_mean_conf", "ocr_low_conf_ratio", "native_text_ratio", "skew_abs_mean", "sharpness",
                  "words_per_page", "n_pages", "mean_field_confidence", "review_share"]


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.linspace(min(ref.min(), cur.min()), max(ref.max(), cur.max()) + 1e-9, bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def report(ref_rows: list[dict], cur_rows: list[dict], alert: float) -> dict:
    out = {}
    for f in DRIFT_FEATURES:
        r = np.array([x[f] for x in ref_rows])
        c = np.array([x[f] for x in cur_rows])
        v = psi(r, c)
        out[f] = {"psi": round(v, 4), "ref_mean": round(float(r.mean()), 4), "cur_mean": round(float(c.mean()), 4),
                  "alert": v >= alert}
    return {"features": out, "alerts": [f for f, v in out.items() if v["alert"]]}
