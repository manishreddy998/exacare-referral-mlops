from referral_mlops.evaluate import score
from referral_mlops.extract.common import FieldResult


def test_precision_and_recall_for_list_and_scalar_fields():
    gold = {"patient_name": "Ann Lee", "diagnoses": ["I10", "E11.9"], "medications": [], "allergies": []}
    pred = {"patient_name": [FieldResult("ann lee", 0.9)],
            "diagnoses": [FieldResult("I10", 1), FieldResult("J18.9", 1)]}
    s = score([pred], [gold])
    d = s["fields"]["diagnoses"]
    assert (d["tp"], d["fp"], d["fn"]) == (1, 1, 1)
    assert s["fields"]["patient_name"]["f1"] == 1.0
    assert s["packet_exact_match"] == 0.0
