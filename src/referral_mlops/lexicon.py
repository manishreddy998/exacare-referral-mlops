"""Reference vocabularies shared by the generator and the extractors.

ICD-10-CM codes and drug names are public reference data. Patients are synthetic.
"""

# code -> (official short description, informal phrasings seen in discharge notes)
ICD10 = {
    "I50.9": ("Heart failure, unspecified", ["CHF", "congestive heart failure", "heart failure"]),
    "I10": ("Essential (primary) hypertension", ["HTN", "hypertension", "high blood pressure"]),
    "E11.9": ("Type 2 diabetes mellitus without complications", ["DM2", "type 2 diabetes", "T2DM"]),
    "J44.1": ("Chronic obstructive pulmonary disease with (acute) exacerbation", ["COPD exacerbation", "AECOPD"]),
    "N18.4": ("Chronic kidney disease, stage 4 (severe)", ["CKD stage 4", "CKD 4"]),
    "I63.9": ("Cerebral infarction, unspecified", ["CVA", "stroke", "acute ischemic stroke"]),
    "S72.001A": ("Fracture of unspecified part of neck of right femur, initial encounter", ["right hip fracture", "R hip fx"]),
    "G30.9": ("Alzheimer's disease, unspecified", ["Alzheimer's dementia", "Alzheimer disease"]),
    "J18.9": ("Pneumonia, unspecified organism", ["pneumonia", "PNA", "community acquired pneumonia"]),
    "N39.0": ("Urinary tract infection, site not specified", ["UTI", "urinary tract infection"]),
    "L89.154": ("Pressure ulcer of sacral region, stage 4", ["stage 4 sacral pressure injury", "sacral decubitus stage IV"]),
    "I48.91": ("Unspecified atrial fibrillation", ["AFib", "atrial fibrillation", "A-fib"]),
    "M17.11": ("Unilateral primary osteoarthritis, right knee", ["right knee OA", "R knee osteoarthritis"]),
    "Z96.651": ("Presence of right artificial knee joint", ["s/p right TKA", "right total knee replacement"]),
    "A41.9": ("Sepsis, unspecified organism", ["sepsis"]),
    "D64.9": ("Anemia, unspecified", ["anemia"]),
    "E87.1": ("Hypo-osmolality and hyponatremia", ["hyponatremia"]),
    "R26.81": ("Unsteadiness on feet", ["gait instability", "unsteady gait"]),
    "F32.A": ("Depression, unspecified", ["depression"]),
    "Z79.4": ("Long term (current) use of insulin", ["insulin dependent", "on long term insulin"]),
    "E66.9": ("Obesity, unspecified", ["obesity"]),
    "Z99.11": ("Dependence on respirator [ventilator] status", ["ventilator dependent", "vent dependent"]),
}

# canonical drug name -> list of (dose, route, frequency) options and brand aliases
MEDICATIONS = {
    "furosemide": (["20 mg PO daily", "40 mg PO BID"], ["Lasix"]),
    "metoprolol tartrate": (["25 mg PO BID", "50 mg PO BID"], ["Lopressor"]),
    "lisinopril": (["10 mg PO daily", "20 mg PO daily"], ["Zestril"]),
    "apixaban": (["5 mg PO BID", "2.5 mg PO BID"], ["Eliquis"]),
    "insulin glargine": (["10 units SC nightly", "18 units SC nightly"], ["Lantus"]),
    "metformin": (["500 mg PO BID", "1000 mg PO BID"], ["Glucophage"]),
    "atorvastatin": (["40 mg PO nightly", "20 mg PO nightly"], ["Lipitor"]),
    "donepezil": (["10 mg PO nightly", "5 mg PO nightly"], ["Aricept"]),
    "acetaminophen": (["650 mg PO q6h PRN", "1000 mg PO TID"], ["Tylenol"]),
    "oxycodone": (["5 mg PO q4h PRN", "10 mg PO q6h PRN"], ["Roxicodone"]),
    "enoxaparin": (["40 mg SC daily", "30 mg SC BID"], ["Lovenox"]),
    "pantoprazole": (["40 mg PO daily"], ["Protonix"]),
    "levothyroxine": (["50 mcg PO daily", "75 mcg PO daily"], ["Synthroid"]),
    "amlodipine": (["5 mg PO daily", "10 mg PO daily"], ["Norvasc"]),
    "sertraline": (["50 mg PO daily", "100 mg PO daily"], ["Zoloft"]),
    "gabapentin": (["300 mg PO TID", "100 mg PO TID"], ["Neurontin"]),
    "tamsulosin": (["0.4 mg PO nightly"], ["Flomax"]),
    "warfarin": (["5 mg PO daily", "2.5 mg PO daily"], ["Coumadin"]),
    "albuterol": (["2 puffs INH q4h PRN"], ["ProAir"]),
    "prednisone": (["40 mg PO daily x5 days", "10 mg PO daily"], ["Deltasone"]),
    "vancomycin": (["1250 mg IV q12h", "1 g IV q12h"], ["Vancocin"]),
    "ceftriaxone": (["1 g IV daily", "2 g IV daily"], ["Rocephin"]),
    "carvedilol": (["6.25 mg PO BID", "12.5 mg PO BID"], ["Coreg"]),
    "senna": (["8.6 mg PO nightly"], ["Senokot"]),
    "docusate": (["100 mg PO BID"], ["Colace"]),
}

ALLERGIES = ["penicillin", "sulfa", "codeine", "latex", "morphine", "iodine", "aspirin", "shellfish"]

# canonical payer -> aliases as they appear on face sheets and fax covers
PAYERS = {
    "Medicare Part A": ["Medicare Part A", "MEDICARE A", "Medicare (Part A)", "Traditional Medicare"],
    "Humana Gold Plus HMO": ["Humana Gold Plus HMO", "HUMANA GOLD PLUS", "Humana MA HMO"],
    "UnitedHealthcare Medicare Advantage": ["UnitedHealthcare Medicare Advantage", "UHC Medicare Advantage", "UHC MA"],
    "Aetna Medicare Advantage": ["Aetna Medicare Advantage", "AETNA MEDICARE", "Aetna MA PPO"],
    "Medicaid": ["Medicaid", "State Medicaid", "MEDICAID"],
    "Blue Cross Blue Shield": ["Blue Cross Blue Shield", "BCBS", "BlueCross BlueShield"],
}

SERVICES = {
    "skilled nursing": ["Skilled Nursing Facility (SNF)", "SNF placement", "Skilled nursing / short term rehab", "SNF"],
    "home health": ["Home Health", "Home health services (PT/OT/RN)", "HHA referral"],
    "hospice": ["Hospice", "Hospice evaluation and admission", "Hospice care"],
    "long-term care": ["Long-Term Care", "LTC placement", "Long term care"],
}

HOSPITALS = [
    "Lakeview General Hospital", "St. Brigid Medical Center", "Riverside Regional Health",
    "Northgate University Hospital", "Mercy Valley Hospital", "Harborpoint Medical Center",
]
