import numpy as np

from referral_mlops.drift import psi
from referral_mlops.extract.common import FieldResult
from referral_mlops.review import choose_threshold, needs_review, route


def test_hard_guards_override_confidence():
    assert needs_review(0.99, {"valid": 0, "ocr_conf": 0.95}, 0.5, 0.8)
    assert needs_review(0.99, {"valid": 1, "ocr_conf": 0.60}, 0.5, 0.8)
    assert not needs_review(0.99, {"valid": 1, "ocr_conf": 0.95}, 0.5, 0.8)


def test_threshold_catches_target_share_of_errors_within_budget():
    gold = {"diagnoses": ["I10"] * 1}
    preds, golds = [], []
    for i in range(100):
        wrong = i < 10
        preds.append({"diagnoses": [FieldResult("J18.9" if wrong else "I10", 0.3 if wrong else 0.95)]})
        golds.append(gold)
    t = choose_threshold(preds, golds, 0.9, 0.5)
    assert 0.3 < t <= 0.95


def test_route_sends_missing_fields_and_unreadable_packets_to_review():
    r = route({"dob": [], "sex": [FieldResult("F", 0.99, {"valid": 1, "ocr_conf": 0.9})]}, [95, 30], 0.5, 50, 0.8)
    assert r["fields"]["dob"][0]["decision"] == "review"
    assert r["fields"]["sex"][0]["decision"] == "auto"
    assert r["packet_review"] and r["unreadable_pages"] == [1]


def test_psi_is_small_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    a, b = rng.normal(90, 3, 500), rng.normal(90, 3, 500)
    assert psi(a, b) < 0.1
    assert psi(a, rng.normal(80, 6, 500)) > 0.2
