import json

import pymupdf

from referral_mlops.synth.patients import make_cohort


def test_cohort_is_deterministic_and_has_no_render_context():
    a, b = make_cohort(5, 3), make_cohort(5, 3)
    assert [r.answer_key() for r in a] == [r.answer_key() for r in b]
    assert "extras" not in a[0].answer_key()


def test_discontinued_meds_are_not_in_the_answer_key():
    for r in make_cohort(50, 9):
        assert not set(r.extras["discontinued"]) & set(r.medications)


def test_packets_render_with_manifest(packet_dir):
    rows = [json.loads(l) for l in (packet_dir / "manifest.jsonl").open()]
    assert len(rows) == 6
    for r in rows:
        doc = pymupdf.open(packet_dir / r["path"])
        assert doc.page_count >= 3
        has_text = any(len(p.get_text().strip()) > 40 for p in doc)
        assert has_text == (r["capture"] == "digital")
