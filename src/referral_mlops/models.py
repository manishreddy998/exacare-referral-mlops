"""Model versions: an extractor, an OCR setup and an optional trained confidence calibrator."""
from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from referral_mlops.extract.common import ALL_FIELDS, FieldResult
from referral_mlops.ocr import OcrConfig, PacketText

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "models"
FEATURES = ["ocr_conf", "vote_share", "sources", "match", "valid", "scanned", "min_page_conf", "heuristic"]


@dataclass
class ModelSpec:
    name: str
    description: str
    extractor: str
    ocr: dict
    calibrate: bool

    @classmethod
    def load(cls, name: str, config_dir: Path = CONFIG_DIR) -> "ModelSpec":
        return cls(**yaml.safe_load((config_dir / f"{name}.yaml").read_text()))

    @property
    def ocr_config(self) -> OcrConfig:
        return OcrConfig(**self.ocr)

    @property
    def ocr_key(self) -> str:
        o = self.ocr_config
        return f"dpi{o.dpi}-rot{int(o.fix_rotation)}-desk{int(o.deskew)}-den{int(o.denoise)}"

    def module(self):
        return importlib.import_module(f"referral_mlops.extract.{self.extractor}")


def feature_row(field: str, r: FieldResult, pt: PacketText) -> list[float]:
    f = r.features
    scanned = 1.0 - pt.features.get("native_text_ratio", 1.0)
    page_confs = pt.features.get("page_confs") or [100.0]
    return [f.get("ocr_conf", 1.0), f.get("votes", 1) / max(f.get("sources", 1), 1), f.get("sources", 1),
            f.get("match", 1.0), f.get("valid", 1), scanned, min(page_confs) / 100, r.confidence] + \
        [1.0 if field == x else 0.0 for x in ALL_FIELDS]


def run_extractor(spec: ModelSpec, pt: PacketText, calibrator=None) -> tuple[dict, float]:
    t0 = time.perf_counter()
    pred = spec.module().extract(pt)
    if calibrator is not None:
        rows, refs = [], []
        for field, results in pred.items():
            for r in results:
                rows.append(feature_row(field, r, pt))
                refs.append(r)
        if rows:
            probs = calibrator.predict_proba(np.array(rows))[:, 1]
            for r, p in zip(refs, probs):
                r.confidence = float(p)
    return pred, time.perf_counter() - t0
