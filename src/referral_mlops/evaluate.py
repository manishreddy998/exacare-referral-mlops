"""Field level precision and recall against the answer key."""
from __future__ import annotations

from collections import defaultdict

from referral_mlops.extract.common import ALL_FIELDS, LIST_FIELDS, normalize


def item_labels(pred: dict, gold: dict) -> list[dict]:
    """One row per predicted value with whether it is correct. Used for scoring and for calibration."""
    rows = []
    for f in ALL_FIELDS:
        g = normalize(f, gold.get(f))
        gset = set(g) if f in LIST_FIELDS else ({g} if g else set())
        for r in pred.get(f, []):
            v = normalize(f, [r.value])[0] if f in LIST_FIELDS else normalize(f, r.value)
            rows.append({"field": f, "value": v, "confidence": r.confidence, "correct": v in gset, **r.features})
    return rows


def score(preds: list[dict], golds: list[dict]) -> dict:
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    perfect = 0
    for pred, gold in zip(preds, golds):
        ok = True
        for f in ALL_FIELDS:
            g = normalize(f, gold.get(f))
            gset = set(g) if f in LIST_FIELDS else ({g} if g else set())
            pvals = [r.value for r in pred.get(f, [])]
            p = normalize(f, pvals) if f in LIST_FIELDS else ({normalize(f, pvals[0])} if pvals else set())
            pset = set(p)
            tp[f] += len(pset & gset)
            fp[f] += len(pset - gset)
            fn[f] += len(gset - pset)
            ok &= pset == gset
        perfect += ok
    out = {"fields": {}}
    for f in ALL_FIELDS:
        prec = tp[f] / (tp[f] + fp[f]) if tp[f] + fp[f] else 0.0
        rec = tp[f] / (tp[f] + fn[f]) if tp[f] + fn[f] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out["fields"][f] = {"precision": prec, "recall": rec, "f1": f1, "tp": tp[f], "fp": fp[f], "fn": fn[f]}
    T, P, N = sum(tp.values()), sum(fp.values()), sum(fn.values())
    out["micro_precision"] = T / (T + P) if T + P else 0.0
    out["micro_recall"] = T / (T + N) if T + N else 0.0
    mp, mr = out["micro_precision"], out["micro_recall"]
    out["micro_f1"] = 2 * mp * mr / (mp + mr) if mp + mr else 0.0
    out["min_field_f1"] = min(v["f1"] for v in out["fields"].values())
    out["packet_exact_match"] = perfect / max(len(golds), 1)
    return out
