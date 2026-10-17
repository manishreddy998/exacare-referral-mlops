from referral_mlops.extract.common import ALL_FIELDS
from referral_mlops.pipeline import gate_decision

G = {"max_micro_f1_drop": 0.005, "max_field_f1_drop": 0.02, "max_p95_seconds": 20,
     "max_cost_increase": 0.25, "min_gain_for_cost": 0.01}


def ev(name, f1, sec, field_f1=None):
    return {"model": name, "metrics": {"micro_f1": f1, "fields": {f: {"f1": (field_f1 or {}).get(f, f1)}
                                                                   for f in ALL_FIELDS}},
            "cost": {"seconds_per_packet_mean": sec, "seconds_per_packet_p95": sec * 2}}


def test_better_candidate_passes_even_if_slower():
    assert gate_decision(ev("prod", 0.76, 1.6), ev("cand", 0.99, 3.1), G)["passed"]


def test_worse_candidate_is_blocked():
    d = gate_decision(ev("prod", 0.99, 3.1), ev("cand", 0.88, 1.6), G)
    assert not d["passed"] and "micro F1" in d["reasons"][0]


def test_one_field_regression_blocks_even_when_average_holds():
    d = gate_decision(ev("prod", 0.99, 3.0), ev("cand", 0.99, 3.0, {"dob": 0.90}), G)
    assert not d["passed"] and any("dob" in r for r in d["reasons"])


def test_cost_increase_without_gain_is_blocked():
    d = gate_decision(ev("prod", 0.99, 2.0), ev("cand", 0.992, 3.0), G)
    assert not d["passed"] and "cost" in d["reasons"][0]
