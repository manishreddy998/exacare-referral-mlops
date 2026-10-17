"""Section aware extractor with OCR repair, vocabulary validation and cross-page voting."""
from __future__ import annotations

import re
from collections import defaultdict

from rapidfuzz import fuzz, process

from referral_mlops.extract.common import FieldResult, clean, norm_person, parse_date
from referral_mlops.lexicon import ALLERGIES, HOSPITALS, ICD10, MEDICATIONS, PAYERS, SERVICES
from referral_mlops.ocr import PacketText

SYNONYMS = {
    "patient_name": ["patient name", "name", "pt", "patient"],
    "dob": ["date of birth", "dob", "d.o.b."],
    "sex": ["sex", "gender"],
    "payer": ["primary insurance", "payer", "ins", "insurance"],
    "member_id": ["member id", "policy #", "subscriber id", "policy"],
    "referring_provider": ["attending physician", "discharging physician", "referring md", "referring physician"],
    "service_requested": ["requested service", "disposition", "service"],
    "allergies": ["allergies"],
}
_key = lambda s: re.sub(r"[^a-z#]", "", s.lower())
SYN_KEYS = {f: [_key(s) for s in syns] for f, syns in SYNONYMS.items()}

PAYER_ALIASES = {a.lower(): p for p, al in PAYERS.items() for a in al}
SERVICE_ALIASES = {a.lower(): s for s, al in SERVICES.items() for a in al}
DRUG_NAMES = {m: m for m in MEDICATIONS} | {b.lower(): m for m, (_, bs) in MEDICATIONS.items() for b in bs}
PHRASES = {}
for code, (desc, informal) in ICD10.items():
    for p in [desc] + informal:
        PHRASES[p.lower()] = code
PHRASE_RE = sorted(PHRASES, key=len, reverse=True)
NEGATION = re.compile(r"family history|mother with|father with|ruled out|was negative|workup for", re.I)
CODE_TOKEN = re.compile(r"\b[A-Z0-9][0-9OIlSZ][0-9A-Z](?:\.[0-9A-Z]{1,4})?\b")
FIRST_CHAR_FIX = {"1": "IL", "2": "Z", "5": "S", "8": "B", "0": "DO", "l": "IL", "I": "IL", "L": "LI"}
DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "I": "1", "i": "1", "l": "1", "S": "5", "B": "8", "Z": "2"})
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DOSE = re.compile(r"\d\s*(mg|mcg|units?|puffs?|g)\b", re.I)


def _split_label(text: str):
    m = re.match(r"^\s*([A-Za-z][A-Za-z .#@]{0,28}?)\s*[:;]\s*(.*)$", text)
    if m:
        return m.group(1), m.group(2)
    return None, None


def _label_field(label: str):
    k = _key(label)
    if "secondary" in label.lower() or "emergency" in label.lower():
        return None, 0
    best, score = None, 0
    for f, keys in SYN_KEYS.items():
        for sk in keys:
            s = 100 if k == sk else (fuzz.ratio(k, sk) if len(k) > 3 and len(sk) > 3 else 0)
            if s > score:
                best, score = f, s
    return (best, score) if score >= 85 else (None, 0)


def _fix_code(tok: str):
    if tok in ICD10:
        return tok, 1.0
    for first in FIRST_CHAR_FIX.get(tok[0], tok[0]):
        t = first + tok[1:3].translate(DIGIT_FIX) + tok[3:]
        if t in ICD10:
            return t, 0.85
        if "." not in t and len(t) > 3 and t[:3] + "." + t[3:] in ICD10:
            return t[:3] + "." + t[3:], 0.8
    return None, 0.0


def _repair_member(v: str):
    s = re.sub(r"[^A-Za-z0-9]", "", v).upper()
    if len(s) < 6:
        return None, 0
    s = s[0] + s[1:].translate(DIGIT_FIX)
    return s, 1 if re.fullmatch(r"[A-Z]\d{9}", s) else 0


def _repair_date(v: str):
    d = parse_date(v)
    if d:
        return d, 1
    v2 = re.sub(r"(?<=[A-Za-z]{3})\s*[.,]?\s*(\d{1,2})[.,\s]+(\d{4})", r" \1 \2", v)
    v2 = re.sub(r"[\[\]|!](?=\d)", "1", v2)
    v2 = re.sub(r"\b[A-Za-z]{3}\b", lambda m: (process.extractOne(m.group(0).lower(), MONTHS, scorer=fuzz.ratio,
                                                                score_cutoff=60) or [m.group(0)])[0].title(), v2)
    v2 = re.sub(r"(?<=\d)[Oo]|[Oo](?=\d)", "0", v2)
    v2 = re.sub(r"(?<=\d)[Il]|[Il](?=\d)", "1", v2)
    m = re.search(r"\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2}|[A-Za-z]{3} \d{1,2} \d{4}|\d{2}-[A-Za-z]{3}-\d{4}", v2)
    d = parse_date(m.group(0)) if m else None
    return (d, 1) if d else (None, 0)


def _order_known(raw: str) -> int:
    """1 when the raw text tells us which token is the family name (a comma, or mixed case)."""
    return int(bool(re.search(r"[A-Za-z]{2}[,.]\s", raw)) or raw != raw.upper())


def _vote(cands: list[tuple]) -> list[FieldResult]:
    """cands: (value, ocr_conf, match, valid[, order_known]). Picks the value most sources agree on.

    Person names are grouped by their set of tokens, so "moreno joseph" read from an all caps line without
    its comma votes together with "joseph moreno", and the spelling comes from a source whose order is known.
    """
    cands = [c for c in cands if c[0]]
    if not cands:
        return []
    groups = defaultdict(list)
    for c in cands:
        groups[frozenset(c[0].split()) if len(c) > 4 else c[0]].append(c)
    _, members = max(groups.items(), key=lambda kv: (len(kv[1]), max(m[1] for m in kv[1])))
    value = max(members, key=lambda m: (m[4] if len(m) > 4 else 0, m[1]))[0]
    ocr = max(m[1] for m in members)
    match = max(m[2] for m in members)
    valid = max(m[3] for m in members)
    feats = {"ocr_conf": ocr, "votes": len(members), "sources": len(cands), "match": match, "valid": valid}
    conf = ocr * match * (len(members) / len(cands)) * (1.0 if valid else 0.6)
    return [FieldResult(value, conf, feats)]


def _fuzzy_vocab(v: str, vocab: dict, cutoff=80):
    hit = process.extractOne(v.lower(), list(vocab), scorer=fuzz.WRatio, score_cutoff=cutoff)
    return (vocab[hit[0]], hit[1] / 100) if hit else (None, 0)


def extract(pt: PacketText) -> dict[str, list[FieldResult]]:
    cands: dict[str, list] = defaultdict(list)
    allergy_lines = []
    lines = pt.lines
    for idx, line in enumerate(lines):
        label, value = _split_label(line.text)
        if label is not None and not value.strip() and idx + 1 < len(lines):
            nxt = lines[idx + 1]
            if nxt.page == line.page and _split_label(nxt.text)[0] is None:
                value = nxt.text
        if label is None:
            continue
        f, score = _label_field(label)
        if not f:
            if label.strip().upper() == "RE":
                m = re.search(r"referral for (.+)", value, re.I)
                if m:
                    cands["patient_name"].append((norm_person(m.group(1)), line.conf / 100, 0.9, 1,
                                                  _order_known(m.group(1))))
            continue
        value = clean(value)
        conf, match = line.conf / 100, score / 100
        if f == "patient_name":
            n = norm_person(value)
            cands[f].append((n, conf, match, int(bool(n and len(n.split()) >= 2)), _order_known(value)))
        elif f == "referring_provider":
            n = norm_person(value)
            cands[f].append((n, conf, match, int(bool(n and len(n.split()) >= 2))))
        elif f == "dob":
            d, ok = _repair_date(value)
            cands[f].append((d, conf, match, ok))
        elif f == "sex":
            v = value.strip().upper()[:1]
            cands[f].append((v if v in "FM" else None, conf, match, 1))
        elif f == "payer":
            p, s = _fuzzy_vocab(value, PAYER_ALIASES)
            cands[f].append((p, conf, match * s, int(p is not None)))
        elif f == "member_id":
            mid, ok = _repair_member(value)
            cands[f].append((mid, conf, match, ok))
        elif f == "service_requested":
            s_, s = _fuzzy_vocab(value, SERVICE_ALIASES)
            cands[f].append((s_, conf, match * s, int(s_ is not None)))
        elif f == "allergies":
            allergy_lines.append((value, conf))

    # referral source directory lookup: any line that names a known sending facility
    for line in pt.lines:
        if len(line.text) < 60:
            hit = process.extractOne(line.text.lower(), [h.lower() for h in HOSPITALS],
                                     scorer=fuzz.partial_ratio, score_cutoff=90)
            if hit:
                cands["referring_facility"].append((HOSPITALS[hit[2]], line.conf / 100, hit[1] / 100, 1))
    out = {f: _vote(cands[f]) for f in
           ["patient_name", "dob", "sex", "payer", "member_id", "referring_provider", "referring_facility",
            "service_requested"]}
    out["diagnoses"] = _diagnoses(pt)
    out["medications"] = _medications(pt)
    out["allergies"] = _allergies(allergy_lines)
    return out


def _diagnoses(pt: PacketText) -> list[FieldResult]:
    found: dict[str, dict] = {}
    by_page = defaultdict(list)
    for l in pt.lines:
        by_page[l.page].append(l)
    for lines in by_page.values():
        text = " ".join(l.text for l in lines)
        conf = sum(l.conf for l in lines) / max(len(lines), 1) / 100
        for sent in re.split(r"(?<=\.)\s+", text):
            if NEGATION.search(sent):
                continue
            for tok in CODE_TOKEN.findall(sent):
                code, s = _fix_code(tok)
                if code:
                    f = found.setdefault(code, {"ocr_conf": conf, "votes": 0, "sources": 1, "match": s, "valid": 1})
                    f["votes"] += 1
                    f["match"] = max(f["match"], s)
            low = sent.lower()
            for ph in PHRASE_RE:
                if re.search(rf"(?<![a-z]){re.escape(ph)}(?![a-z])", low):
                    code = PHRASES[ph]
                    f = found.setdefault(code, {"ocr_conf": conf, "votes": 0, "sources": 1, "match": 0.9, "valid": 1})
                    f["votes"] += 1
    return [FieldResult(c, f["ocr_conf"] * f["match"], f) for c, f in sorted(found.items())]


def _medications(pt: PacketText) -> list[FieldResult]:
    found: dict[str, dict] = {}
    skip, page = False, None
    names = list(DRUG_NAMES)
    for l in pt.lines:
        if l.page != page:
            skip, page = False, l.page
        low = l.text.lower()
        if fuzz.partial_ratio("discontinued", low) >= 85 and len(low) > 10:
            skip = True
            continue
        if fuzz.partial_ratio("allergies", low) >= 85:
            skip = False
            continue
        if skip or not DOSE.search(l.text):
            continue
        head = re.sub(r"^[^A-Za-z]+", "", l.text)
        words = head.split()
        for n in (2, 1):
            hit = process.extractOne(" ".join(words[:n]).lower(), names, scorer=fuzz.ratio, score_cutoff=85)
            if hit:
                med = DRUG_NAMES[hit[0]]
                f = found.setdefault(med, {"ocr_conf": l.conf / 100, "votes": 0, "sources": 1,
                                           "match": hit[1] / 100, "valid": 1})
                f["votes"] += 1
                break
    return [FieldResult(m, f["ocr_conf"] * f["match"], f) for m, f in sorted(found.items())]


def _allergies(lines) -> list[FieldResult]:
    out = {}
    for value, conf in lines:
        low = value.lower()
        if re.search(r"nkda|no known", low):
            continue
        for tok in re.split(r"[,()\s]+", low):
            hit = process.extractOne(tok, ALLERGIES, scorer=fuzz.ratio, score_cutoff=80) if len(tok) > 3 else None
            if hit:
                out[hit[0]] = FieldResult(hit[0], conf * hit[1] / 100,
                                          {"ocr_conf": conf, "votes": 1, "sources": 1, "match": hit[1] / 100, "valid": 1})
    return list(out.values())
