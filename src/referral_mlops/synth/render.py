"""Render a synthetic referral into a multi-page PDF packet in one of several layouts."""
from __future__ import annotations

import random
import textwrap
from datetime import date
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from referral_mlops.lexicon import ICD10, MEDICATIONS
from referral_mlops.synth.patients import Referral

LAYOUTS = ["hospital_portal", "discharge_summary", "e_referral"]
W, H = letter


def _name_variant(first: str, last: str, rng: random.Random) -> str:
    return rng.choice([f"{last.upper()}, {first.upper()}", f"{first} {last}", f"{last}, {first}"])


def _dob_variant(iso: str, rng: random.Random) -> str:
    d = date.fromisoformat(iso)
    return rng.choice([
        d.strftime("%m/%d/%Y"), d.isoformat(), d.strftime("%b %d %Y"), d.strftime("%d-%b-%Y").upper(),
    ])


def _provider_variant(name: str, rng: random.Random) -> str:
    first, last = name.split(" ", 1)
    return rng.choice([f"Dr. {first} {last}, MD", f"{last.upper()}, {first.upper()} MD", f"{first} {last}, DO"])


def _member_variant(mid: str, rng: random.Random) -> str:
    if rng.random() < 0.25:
        return f"{mid[:4]} {mid[4:7]} {mid[7:]}"
    return mid


def _med_line(name: str, rng: random.Random) -> tuple[str, str]:
    doses, brands = MEDICATIONS[name]
    shown = brands[0] if rng.random() < 0.2 else name
    if rng.random() < 0.5:
        shown = shown.capitalize()
    return shown, rng.choice(doses)


def _informal(code: str, rng: random.Random) -> str:
    return rng.choice(ICD10[code][1])


class Page:
    def __init__(self, kind: str):
        self.kind = kind
        self.ops: list[tuple] = []

    def title(self, t):
        self.ops.append(("title", t))

    def kv(self, k, v):
        self.ops.append(("kv", k, v))

    def text(self, t):
        self.ops.append(("text", t))

    def table(self, headers, rows):
        self.ops.append(("table", headers, rows))

    def gap(self):
        self.ops.append(("gap",))


def _fax_cover(r: Referral, rng: random.Random, n_pages: int) -> Page:
    e = r.extras
    p = Page("fax_cover")
    p.title("FAX TRANSMITTAL")
    p.kv("FROM", f"{r.referring_facility} Case Management")
    p.kv("PHONE", e["hospital_phone"])
    p.kv("TO", "Admissions Department")
    p.kv("PAGES", f"{n_pages} (including cover)")
    p.kv("RE", f"Referral for {_name_variant(e['first'], e['last'], rng)}")
    p.kv("Contact", e["case_manager"])
    p.gap()
    p.text("CONFIDENTIALITY NOTICE: This facsimile contains protected health information intended only for the "
           "named recipient. If you received this in error, notify the sender and destroy all copies.")
    return p


def _demographics(r: Referral, rng: random.Random, layout: str) -> Page:
    e = r.extras
    p = Page("face_sheet")
    p.text(r.referring_facility.upper())
    labels = {
        "hospital_portal": ("Patient Name", "Date of Birth", "Sex", "Primary Insurance", "Member ID"),
        "discharge_summary": ("NAME", "DOB", "GENDER", "Payer", "Policy #"),
        "e_referral": ("Pt", "D.O.B.", "Sex", "Ins", "Subscriber ID"),
    }[layout]
    p.title({"hospital_portal": "PATIENT FACE SHEET", "discharge_summary": "ADMISSION RECORD",
             "e_referral": "eReferral: Patient Demographics"}[layout])
    p.kv(labels[0], _name_variant(e["first"], e["last"], rng))
    p.kv(labels[1], _dob_variant(r.dob, rng))
    p.kv(labels[2], {"F": rng.choice(["F", "Female"]), "M": rng.choice(["M", "Male"])}[r.sex])
    p.kv("MRN", e["mrn"])
    p.kv(labels[3], e["payer_alias"])
    p.kv(labels[4], _member_variant(r.member_id, rng))
    if e["secondary_payer"]:
        p.kv("Secondary Insurance", e["secondary_payer"])
    p.kv("Emergency Contact", f"{e['emergency_contact']} (family)")
    p.kv("Admit Date", e["admit_date"])
    return p


def _clinical(r: Referral, rng: random.Random, layout: str) -> Page:
    e = r.extras
    p = Page("clinical")
    p.text(r.referring_facility)
    uncoded = set(e["uncoded"])
    prov = _provider_variant(r.referring_provider, rng)
    if layout == "hospital_portal":
        p.title("HISTORY AND PHYSICAL")
        p.kv("Attending Physician", prov)
        p.kv("Requested Service", e["service_alias"])
        p.gap()
        p.text("Diagnoses:")
        for c in r.diagnoses:
            p.text(f"  {ICD10[c][0]}" if c in uncoded else f"  {c}  {ICD10[c][0]}")
    elif layout == "discharge_summary":
        p.title("DISCHARGE SUMMARY")
        p.kv("Discharging physician", prov)
        p.kv("Disposition", e["service_alias"])
        p.gap()
        main, rest = r.diagnoses[0], r.diagnoses[1:]
        fmt = lambda c: _informal(c, rng) if c in uncoded else f"{_informal(c, rng)} ({c})"
        p.text(f"Principal diagnosis: {fmt(main)}.")
        if rest:
            p.text("Secondary diagnoses: " + "; ".join(fmt(c) for c in rest) + ".")
    else:
        p.title("eReferral: Clinical Summary")
        p.kv("Referring MD", prov)
        p.kv("Service", e["service_alias"])
        coded = [c for c in r.diagnoses if c not in uncoded]
        if coded:
            p.kv("Dx codes", ", ".join(coded))
        if uncoded:
            p.kv("Problem list", ", ".join(_informal(c, rng) for c in uncoded))
    p.gap()
    hpi = (f"Patient is a {2026 - int(r.dob[:4])} year old {'female' if r.sex == 'F' else 'male'} admitted on "
           f"{e['admit_date']} and now medically stable for discharge. Seen by {e['nurse']} on the unit. ")
    if e["family_history_code"]:
        hpi += f"Family history: mother with {_informal(e['family_history_code'], rng)}. "
    if e["ruled_out_code"]:
        hpi += f"Workup for {_informal(e['ruled_out_code'], rng)} was negative; ruled out. "
    hpi += "Needs assistance with ADLs and therapy 5x weekly."
    p.text(hpi)
    return p


def _meds(r: Referral, rng: random.Random, layout: str) -> Page:
    e = r.extras
    p = Page("medications")
    p.title(rng.choice(["MEDICATION RECONCILIATION", "Discharge Medication List", "Active Medications"]))
    rows = [list(_med_line(m, rng)) for m in r.medications]
    if layout == "discharge_summary":
        for name, dose in rows:
            p.text(f"- {name} {dose}")
    else:
        p.table(["Medication", "Dose / Route / Frequency"], rows)
    if e["discontinued"]:
        p.gap()
        p.text("Discontinued during this admission (do NOT continue):")
        for m in e["discontinued"]:
            name, dose = _med_line(m, rng)
            p.text(f"  {name} {dose}")
    p.gap()
    if r.allergies:
        reactions = ["rash", "hives", "anaphylaxis", "nausea"]
        p.kv("Allergies", ", ".join(f"{a.capitalize()} ({rng.choice(reactions)})" for a in r.allergies))
    else:
        p.kv("Allergies", rng.choice(["NKDA", "No known drug allergies"]))
    return p


def _labs(r: Referral, rng: random.Random) -> Page:
    p = Page("labs")
    p.title("LABORATORY RESULTS")
    p.table(["Test", "Result", "Ref range"], [
        ["Sodium", f"{rng.randint(128, 146)} mmol/L", "135-145"],
        ["Creatinine", f"{rng.uniform(0.6, 3.2):.1f} mg/dL", "0.6-1.3"],
        ["Hemoglobin", f"{rng.uniform(8, 15):.1f} g/dL", "12-16"],
        ["WBC", f"{rng.uniform(4, 16):.1f} K/uL", "4.5-11"],
    ])
    p.gap()
    p.kv("Collected by", r.extras["nurse"])
    return p


def build_pages(r: Referral, rng: random.Random, layout: str) -> list[Page]:
    body = [_demographics(r, rng, layout), _clinical(r, rng, layout), _meds(r, rng, layout)]
    if rng.random() < 0.6:
        body.append(_labs(r, rng))
    if rng.random() < 0.5:
        rng.shuffle(body)
    pages = body
    if rng.random() < 0.7:
        pages = [_fax_cover(r, rng, len(body) + 1)] + body
    return pages


def draw_pdf(pages: list[Page], path: Path, rng: random.Random) -> None:
    c = canvas.Canvas(str(path), pagesize=letter)
    font = rng.choice(["Helvetica", "Times-Roman", "Courier"])
    bold = {"Helvetica": "Helvetica-Bold", "Times-Roman": "Times-Bold", "Courier": "Courier-Bold"}[font]
    for page in pages:
        y = H - 60
        x0 = 54 + rng.randint(-8, 8)
        for op in page.ops:
            if op[0] == "title":
                c.setFont(bold, 15)
                c.drawString(x0, y, op[1])
                y -= 28
            elif op[0] == "kv":
                c.setFont(bold, 11)
                c.drawString(x0, y, f"{op[1]}:")
                c.setFont(font, 11)
                c.drawString(x0 + 150, y, str(op[2]))
                y -= 18
            elif op[0] == "text":
                c.setFont(font, 11)
                for line in textwrap.wrap(op[1], 88, subsequent_indent="  " if op[1].startswith("  ") else "") or [""]:
                    c.drawString(x0, y, line)
                    y -= 15
            elif op[0] == "table":
                headers, rows = op[1], op[2]
                cols = [x0, x0 + 200, x0 + 380]
                c.setFont(bold, 11)
                for i, h in enumerate(headers):
                    c.drawString(cols[i], y, h)
                y -= 6
                c.line(x0, y, x0 + 480, y)
                y -= 14
                c.setFont(font, 11)
                for row in rows:
                    for i, cell in enumerate(row):
                        c.drawString(cols[i], y, str(cell))
                    y -= 16
            elif op[0] == "gap":
                y -= 12
        c.setFont(font, 8)
        c.drawString(x0, 30, "Generated synthetic record for testing. Not a real patient.")
        c.showPage()
    c.save()
