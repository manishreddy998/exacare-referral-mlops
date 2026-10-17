"""Turn a clean digital PDF into a scanned or faxed copy with realistic damage."""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import pymupdf as fitz
import numpy as np

PROFILES = {
    # name: dpi, max skew degrees, blur kernel options, speckle rate, binarize, chance of a 90/180 turn
    "scan": dict(dpi=200, skew=2.0, blur=[1, 3], speckle=0.0005, binarize=False, turn=0.0, jpeg=70),
    "fax": dict(dpi=200, skew=2.5, blur=[1, 3], speckle=0.0015, binarize=True, turn=0.08, jpeg=60),
    # used for the drift batch: a new low quality fax source
    "bad_fax": dict(dpi=150, skew=4.0, blur=[3, 5], speckle=0.004, binarize=True, turn=0.15, jpeg=35),
}


def _damage(img: np.ndarray, prof: dict, rng: random.Random, page_no: int, total: int) -> np.ndarray:
    h, w = img.shape[:2]
    angle = rng.uniform(-prof["skew"], prof["skew"])
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, m, (w, h), borderValue=255)
    k = rng.choice(prof["blur"])
    if k > 1:
        img = cv2.GaussianBlur(img, (k, k), 0)
    nprng = np.random.default_rng(rng.randint(0, 2**31))
    noise = nprng.random(img.shape) < prof["speckle"]
    img = img.copy()
    img[noise] = 0
    if prof["binarize"]:
        img = np.where(img > rng.randint(150, 190), 255, 0).astype(np.uint8)
        # horizontal streaks typical of a dirty fax drum
        for _ in range(rng.randint(0, 3)):
            yy = rng.randint(0, h - 3)
            img[yy:yy + rng.randint(1, 2), :] = 0
        header = f"FAX  {rng.randint(10, 12)}:{rng.randint(10, 59)}  P.{page_no:03d}/{total:03d}"
        cv2.putText(img, header, (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 1)
    if rng.random() < prof["turn"]:
        img = cv2.rotate(img, rng.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180]))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, prof["jpeg"]])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def degrade_pdf(src: Path, dst: Path, profile: str, rng: random.Random) -> None:
    prof = PROFILES[profile]
    doc = fitz.open(src)
    out = fitz.open()
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=prof["dpi"], colorspace=fitz.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        img = _damage(img, prof, rng, i + 1, len(doc))
        ok, png = cv2.imencode(".png", img)
        h, w = img.shape
        pw, ph = w * 72 / prof["dpi"], h * 72 / prof["dpi"]
        np_ = out.new_page(width=pw, height=ph)
        np_.insert_image(np_.rect, stream=png.tobytes())
    out.save(dst, deflate=True)
    out.close()
    doc.close()
