import logging

from referral_mlops.phi import PhiFilter, audit, redact


def test_redact_removes_identifiers():
    s = redact("Patient: Tina Smith DOB: 1959-09-26 member A123456789 MRN1234567 (555) 123-4567")
    for leak in ("Tina", "1959", "A123456789", "MRN1234567", "555"):
        assert leak not in s


def test_filter_applies_to_formatted_args(caplog):
    log = logging.getLogger("phi-test")
    log.addFilter(PhiFilter())
    with caplog.at_level(logging.INFO, logger="phi-test"):
        log.info("got %s", "DOB: 1959-09-26")
    assert "1959" not in caplog.text


def test_audit_finds_a_leak(tmp_path):
    (tmp_path / "ok.log").write_text("doc=abc latency_ms=10")
    (tmp_path / "bad.log").write_text("extracted tina smith")
    key = {"patient_name": "Tina Smith", "dob": "1959-09-26", "member_id": "A123456789"}
    res = audit([tmp_path], [key])
    assert not res["passed"] and res["leaks"][0]["file"].endswith("bad.log")
