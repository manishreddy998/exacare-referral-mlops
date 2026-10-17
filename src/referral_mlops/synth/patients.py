"""Synthetic patient referrals with an answer key. No real people, no real PHI."""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

from faker import Faker

from referral_mlops.lexicon import ALLERGIES, HOSPITALS, ICD10, MEDICATIONS, PAYERS, SERVICES


@dataclass
class Referral:
    packet_id: str
    patient_name: str
    dob: str
    sex: str
    payer: str
    member_id: str
    referring_provider: str
    referring_facility: str
    service_requested: str
    diagnoses: list[str]
    medications: list[str]
    allergies: list[str]
    # context used only for rendering, never scored
    extras: dict = field(default_factory=dict)

    def answer_key(self) -> dict:
        d = asdict(self)
        d.pop("extras")
        return d


def make_referral(packet_id: str, rng: random.Random, fake: Faker) -> Referral:
    sex = rng.choice(["F", "M"])
    first = fake.first_name_female() if sex == "F" else fake.first_name_male()
    last = fake.last_name()
    dob = date(1930, 1, 1) + timedelta(days=rng.randint(0, 365 * 35))
    payer = rng.choice(list(PAYERS))
    member_id = f"{rng.choice('ABCDHJKMNRW')}{rng.randint(10**8, 10**9 - 1)}"
    provider = f"{fake.first_name()} {fake.last_name()}"
    codes = rng.sample(list(ICD10), rng.randint(2, 5))
    meds = rng.sample(list(MEDICATIONS), rng.randint(3, 7))
    remaining = [m for m in MEDICATIONS if m not in meds]
    discontinued = rng.sample(remaining, rng.randint(0, 2))
    allergies = rng.sample(ALLERGIES, rng.randint(0, 2))
    others = [c for c in ICD10 if c not in codes]
    extras = {
        "first": first,
        "last": last,
        "mrn": f"MRN{rng.randint(10**6, 10**7 - 1)}",
        "emergency_contact": f"{fake.first_name()} {last if rng.random() < 0.5 else fake.last_name()}",
        "nurse": f"{fake.first_name()} {fake.last_name()}, RN",
        "case_manager": f"{fake.first_name()} {fake.last_name()}",
        "discontinued": discontinued,
        "secondary_payer": rng.choice([None, None, "Medicaid", "Blue Cross Blue Shield"]),
        "family_history_code": rng.choice([None, rng.choice(others)]),
        "ruled_out_code": rng.choice([None, None, rng.choice(others)]),
        "admit_date": (date(2026, 9, 1) + timedelta(days=rng.randint(0, 40))).isoformat(),
        "hospital_phone": fake.numerify("(###) ###-####"),
        "uncoded": [c for c in codes if rng.random() < 0.35],
        "service_alias": rng.choice(SERVICES[(svc := rng.choice(list(SERVICES)))]),
        "payer_alias": rng.choice(PAYERS[payer]),
    }
    if extras["secondary_payer"] == payer:
        extras["secondary_payer"] = None
    return Referral(
        packet_id=packet_id,
        patient_name=f"{first} {last}",
        dob=dob.isoformat(),
        sex=sex,
        payer=payer,
        member_id=member_id,
        referring_provider=provider,
        referring_facility=rng.choice(HOSPITALS),
        service_requested=svc,
        diagnoses=codes,
        medications=meds,
        allergies=allergies,
        extras=extras,
    )


def make_cohort(n: int, seed: int, prefix: str = "PKT") -> list[Referral]:
    rng = random.Random(seed)
    fake = Faker("en_US")
    fake.seed_instance(seed)
    return [make_referral(f"{prefix}-{i:04d}", rng, fake) for i in range(n)]
