import json

from referral_mlops.evaluate import score
from referral_mlops.extract import layout_v2, rules_v1
from referral_mlops.extract.common import norm_person, parse_date
from referral_mlops.extract.layout_v2 import _fix_code, _repair_date
from referral_mlops.ocr import Line, PacketText, read_packet


def test_name_and_date_normalizers():
    assert norm_person("SMITH, TINA") == "tina smith"
    assert norm_person("Dr. Mike Burton, MD") == "mike burton"
    assert norm_person("Moreno. Joseph") == "joseph moreno"
    assert parse_date("26-SEP-1959") == "1959-09-26"
    assert parse_date("Mar 14 1941") == "1941-03-14"


def test_ocr_code_repair_uses_the_vocabulary():
    assert _fix_code("110")[0] == "I10"
    assert _fix_code("279.4")[0] == "Z79.4"
    assert _fix_code("189.154")[0] == "L89.154"
    assert _fix_code("650")[0] is None


def test_date_repair_handles_common_ocr_damage():
    assert _repair_date("Apr 11.1963")[0] == "1963-04-11"
    assert _repair_date("Nov 29 [938")[0] == "1938-11-29"


def _pt(lines):
    return PacketText([Line(t, 95.0, 0) for t in lines], {"native_text_ratio": 0.0, "page_confs": [95.0]})


def test_family_history_and_ruled_out_are_not_diagnoses():
    pt = _pt(["Principal diagnosis: CHF (I50.9).",
              "Family history: mother with stroke. Workup for pneumonia was negative; ruled out."])
    codes = {r.value for r in layout_v2.extract(pt)["diagnoses"]}
    assert codes == {"I50.9"}


def test_discontinued_medications_are_excluded():
    pt = _pt(["Active Medications", "- Lasix 40 mg PO BID", "Discontinued during this admission (do NOT continue):",
              "  Warfarin 5 mg PO daily", "Allergies: NKDA"])
    meds = {r.value for r in layout_v2.extract(pt)["medications"]}
    assert meds == {"furosemide"}
    assert {r.value for r in rules_v1.extract(pt)["medications"]} == {"furosemide", "warfarin"}


def test_layout_v2_on_digital_packets(packet_dir):
    rows = [json.loads(l) for l in (packet_dir / "manifest.jsonl").open()]
    digital = [r for r in rows if r["capture"] == "digital"]
    preds = [layout_v2.extract(read_packet(packet_dir / r["path"])) for r in digital]
    assert score(preds, [r["answer"] for r in digital])["micro_f1"] > 0.97
