"""Shared types and normalizers for extractors and the evaluator."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

SCALAR_FIELDS = [
    "patient_name", "dob", "sex", "payer", "member_id", "referring_provider", "referring_facility", "service_requested",
]
LIST_FIELDS = ["diagnoses", "medications", "allergies"]
ALL_FIELDS = SCALAR_FIELDS + LIST_FIELDS


@dataclass
class Candidate:
    value: str
    conf: float  # OCR confidence of the source line, 0..1
    page: int
    match: float = 1.0  # how cleanly the label or vocabulary matched, 0..1


@dataclass
class FieldResult:
    value: str | None
    confidence: float
    features: dict = field(default_factory=dict)


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip(" .,:;|_-'\"")


def norm_person(s: str | None) -> str | None:
    if not s:
        return None
    s = re.sub(r"^(dr\.?\s+)", "", s.strip(), flags=re.I)
    s = re.sub(r",?\s*\b(md|m\.d|mo|m0|do|d0|rn|np|pa)\b\.?$", "", s, flags=re.I).strip()
    # OCR often turns the comma in "LAST, FIRST" into a period
    s = re.sub(r"^([A-Za-z'-]{2,})\.\s+(?=[A-Za-z])", r"\1, ", s)
    if "," in s:
        last, first = [p.strip() for p in s.split(",", 1)]
        s = f"{first} {last}"
    s = re.sub(r"[^A-Za-z' -]", "", s)
    return clean(s).lower() or None


DATE_FORMATS = ["%m/%d/%Y", "%Y-%m-%d", "%b %d %Y", "%d-%b-%Y", "%m-%d-%Y", "%b %d, %Y"]


def parse_date(s: str | None) -> str | None:
    if not s:
        return None
    s = clean(s)
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s.title() if "%b" in fmt else s, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def norm_member(s: str | None) -> str | None:
    if not s:
        return None
    s = re.sub(r"[^A-Za-z0-9]", "", s).upper()
    return s or None


def normalize(field_name: str, value):
    if value is None:
        return [] if field_name in LIST_FIELDS else None
    if field_name in ("patient_name", "referring_provider"):
        return norm_person(value)
    if field_name == "member_id":
        return norm_member(value)
    if field_name == "referring_facility":
        return clean(str(value)).lower()
    if field_name in LIST_FIELDS:
        return sorted({str(v).strip().lower() for v in value})
    return str(value).strip().lower()
