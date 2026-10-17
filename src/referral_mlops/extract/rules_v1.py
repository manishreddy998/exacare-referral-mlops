"""Baseline extractor: fixed labels and exact vocabulary lookups over the whole packet."""
from __future__ import annotations

import re

from referral_mlops.extract.common import FieldResult, clean, norm_person, parse_date
from referral_mlops.lexicon import ALLERGIES, ICD10, MEDICATIONS, PAYERS, SERVICES
from referral_mlops.ocr import PacketText

LABELS = {
    "patient_name": r"(?:Patient Name|NAME|Pt)",
    "dob": r"(?:Date of Birth|DOB|D\.O\.B\.)",
    "sex": r"(?:Sex|GENDER)",
    "payer": r"(?:Primary Insurance|Payer|Ins)",
    "member_id": r"(?:Member ID|Policy #|Subscriber ID)",
    "referring_provider": r"(?:Attending Physician|Discharging physician|Referring MD)",
    "service_requested": r"(?:Requested Service|Disposition|Service)",
}
ICD_RE = re.compile(r"\b[A-TV-Z][0-9][0-9AB](?:\.[0-9A-Z]{1,4})?\b")


def _first(pt: PacketText, label: str):
    rx = re.compile(rf"^\s*{label}\s*:\s*(.+)$")
    for line in pt.lines:
        m = rx.match(line.text)
        if m:
            return clean(m.group(1)), line.conf / 100
    return None, 0.0


def _one(value, conf):
    return [FieldResult(value, conf, {"ocr_conf": conf, "votes": 1, "sources": 1, "match": 1.0, "valid": 1})] if value else []


def extract(pt: PacketText) -> dict[str, list[FieldResult]]:
    out: dict[str, list[FieldResult]] = {}
    v, c = _first(pt, LABELS["patient_name"])
    out["patient_name"] = _one(norm_person(v), c)
    v, c = _first(pt, LABELS["dob"])
    out["dob"] = _one(parse_date(v), c)
    v, c = _first(pt, LABELS["sex"])
    out["sex"] = _one(v[0].upper() if v else None, c)
    v, c = _first(pt, LABELS["payer"])
    canon = next((p for p, al in PAYERS.items() for a in al if v and a.lower() == v.lower()), None)
    out["payer"] = _one(canon, c)
    v, c = _first(pt, LABELS["member_id"])
    out["member_id"] = _one(re.sub(r"\s", "", v).upper() if v else None, c)
    v, c = _first(pt, LABELS["referring_provider"])
    out["referring_provider"] = _one(norm_person(v), c)
    v, c = _first(pt, LABELS["service_requested"])
    svc = next((s for s, al in SERVICES.items() for a in al if v and a.lower() == v.lower()), None)
    out["service_requested"] = _one(svc, c)

    v, c = _first(pt, r"FROM")
    out["referring_facility"] = _one(re.sub(r"\s*Case Management$", "", v) if v else None, c)

    text = pt.text
    codes = sorted({m for m in ICD_RE.findall(text) if m in ICD10})
    out["diagnoses"] = [FieldResult(cd, 1.0, {"ocr_conf": 1.0, "votes": 1, "sources": 1, "match": 1.0, "valid": 1}) for cd in codes]
    low = text.lower()
    meds = [m for m, (_, brands) in MEDICATIONS.items()
            if re.search(rf"\b{re.escape(m)}\b", low) or any(re.search(rf"\b{b.lower()}\b", low) for b in brands)]
    out["medications"] = [FieldResult(m, 1.0, {"ocr_conf": 1.0, "votes": 1, "sources": 1, "match": 1.0, "valid": 1}) for m in meds]
    v, c = _first(pt, r"Allergies")
    alls = [a for a in ALLERGIES if v and re.search(rf"\b{a}\b", v.lower())]
    out["allergies"] = [FieldResult(a, c, {"ocr_conf": c, "votes": 1, "sources": 1, "match": 1.0, "valid": 1}) for a in alls]
    return out
