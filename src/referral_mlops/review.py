"""Human in the loop routing: auto-accept confident fields, send the rest to a review queue."""
from __future__ import annotations

import numpy as np

from referral_mlops.evaluate import item_labels


def needs_review(confidence: float, features: dict, threshold: float, min_ocr_conf: float) -> bool:
    """Calibrated confidence decides, with two hard guards the calibrator cannot talk its way past:
    a value that fails its format check, or one read from a line the OCR engine itself was unsure about."""
    return (confidence < threshold or not features.get("valid", 1)
            or features.get("ocr_conf", 1.0) < min_ocr_conf)


def choose_threshold(preds, golds, target_error_capture: float, max_review_share: float = 1.0) -> float:
    """Confidence cut for review routing, set from labelled data.

    Picks the smallest cut that sends at least target_error_capture of the known wrong values to a person.
    If that would route more than max_review_share of all values, it takes the largest cut that stays inside
    the review budget instead. A pure precision target is not used because it is met trivially when the
    labelled data is nearly perfect."""
    rows = [r for p, g in zip(preds, golds) for r in item_labels(p, g)]
    conf = np.array([r["confidence"] for r in rows])
    wrong = ~np.array([r["correct"] for r in rows], dtype=bool)
    if not wrong.any():
        return 0.0
    best = 0.0
    for t in np.round(np.arange(0.0, 1.0001, 0.005), 3):
        if (conf < t).mean() > max_review_share:
            break
        best = float(t)
        if (conf[wrong] < t).mean() >= target_error_capture:
            return best
    return best


def route(pred: dict, page_confs: list[float], threshold: float, unreadable_conf: float,
          min_ocr_conf: float = 0.0) -> dict:
    """Returns per field decisions plus a packet level flag for unreadable pages."""
    unreadable = [i for i, c in enumerate(page_confs) if c < unreadable_conf]
    decisions = {}
    for field, results in pred.items():
        decisions[field] = [{"value": r.value, "confidence": round(r.confidence, 3),
                             "decision": "review" if needs_review(r.confidence, r.features, threshold, min_ocr_conf)
                             else "auto"} for r in results]
        if not results:
            decisions[field] = [{"value": None, "confidence": 0.0, "decision": "review"}]
    return {"fields": decisions, "unreadable_pages": unreadable, "packet_review": bool(unreadable)}


def review_stats(preds, golds, texts, threshold: float, unreadable_conf: float, min_ocr_conf: float = 0.0) -> dict:
    auto_ok = auto_n = review_n = review_err = total_err = 0
    missing_total = missing_caught = packets_flagged = 0
    from referral_mlops.evaluate import score
    for pred, gold, pt in zip(preds, golds, texts):
        flagged = any(c < unreadable_conf for c in pt.features.get("page_confs", []))
        packets_flagged += flagged
        for r in item_labels(pred, gold):
            err = not r["correct"]
            total_err += err
            if not flagged and not needs_review(r["confidence"], r, threshold, min_ocr_conf):
                auto_n += 1
                auto_ok += not err
            else:
                review_n += 1
                review_err += err
        s = score([pred], [gold])
        fn = sum(v["fn"] for v in s["fields"].values())
        missing_total += fn
        empty_fields = sum(1 for v in pred.values() if not v)
        missing_caught += fn if (flagged or empty_fields) else 0
    n = auto_n + review_n
    return {
        "threshold": threshold,
        "min_ocr_conf": min_ocr_conf,
        "auto_accept_share": auto_n / n if n else 0.0,
        "auto_accept_precision": auto_ok / auto_n if auto_n else 0.0,
        "review_share": review_n / n if n else 0.0,
        "wrong_values_caught_by_review": review_err / total_err if total_err else 1.0,
        "packets_flagged_unreadable": packets_flagged,
        "missed_values_in_flagged_or_incomplete_packets": missing_caught / missing_total if missing_total else 1.0,
    }
