"""Read text from a referral packet: text layer for digital pages, Tesseract for scanned ones."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytesseract

# Tesseract spawns one thread per core by default; with a process pool that oversubscribes the CPU badly.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")


@dataclass
class OcrConfig:
    dpi: int = 300
    fix_rotation: bool = True
    deskew: bool = True
    denoise: bool = True


@dataclass
class Line:
    text: str
    conf: float  # 0..100, mean word confidence; 100 for a native text layer
    page: int


@dataclass
class PacketText:
    lines: list[Line]
    features: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(l.text for l in self.lines)


def _estimate_skew(img: np.ndarray) -> float:
    small = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    inv = 255 - cv2.medianBlur(small, 3)
    best, best_score = 0.0, -1.0
    h, w = inv.shape
    for a in np.arange(-5, 5.01, 0.25):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        r = cv2.warpAffine(inv, m, (w, h), borderValue=0)
        score = float(np.var(r.sum(axis=1)))
        if score > best_score:
            best, best_score = float(a), score
    return best


def _fix_rotation(img: np.ndarray) -> tuple[np.ndarray, int]:
    try:
        osd = pytesseract.image_to_osd(img, config="--psm 0 -c min_characters_to_try=20")
        angle = int(next(l for l in osd.splitlines() if l.startswith("Rotate:")).split(":")[1])
    except Exception:
        return img, 0
    rot = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    return (cv2.rotate(img, rot[angle]), angle) if angle in rot else (img, 0)


def _ocr_lines(img: np.ndarray, page: int) -> tuple[list[Line], list[float]]:
    data = pytesseract.image_to_data(img, config="--psm 4", output_type=pytesseract.Output.DICT)
    groups: dict[tuple, list[tuple[str, float]]] = {}
    confs = []
    for i, word in enumerate(data["text"]):
        c = float(data["conf"][i])
        if not word.strip() or c < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        groups.setdefault(key, []).append((word, c))
        confs.append(c)
    lines = [Line(" ".join(w for w, _ in ws), float(np.mean([c for _, c in ws])), page)
             for _, ws in sorted(groups.items())]
    return lines, confs


def _native_lines(page) -> list[str]:
    """Rebuild visual lines from positioned words so a label and its value stay together."""
    words = sorted(page.get_text("words"), key=lambda w: (w[1] + w[3]) / 2)
    rows: list[tuple[float, list]] = []
    for x0, y0, x1, y1, word, *_ in words:
        mid = (y0 + y1) / 2
        if rows and abs(rows[-1][0] - mid) < 4:
            rows[-1][1].append((x0, word))
        else:
            rows.append((mid, [(x0, word)]))
    return [" ".join(w for _, w in sorted(ws)) for _, ws in rows]


def read_packet(path: Path, cfg: OcrConfig | None = None) -> PacketText:
    cfg = cfg or OcrConfig()
    t0 = time.perf_counter()
    doc = pymupdf.open(path)
    lines: list[Line] = []
    confs: list[float] = []
    native, skews, turns, blur, page_confs = 0, [], 0, [], []
    for pno, page in enumerate(doc):
        if len(page.get_text().strip()) > 40:
            native += 1
            lines += [Line(t, 100.0, pno) for t in _native_lines(page)]
            page_confs.append(100.0)
            continue
        pix = page.get_pixmap(dpi=cfg.dpi, colorspace=pymupdf.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
        blur.append(float(cv2.Laplacian(img, cv2.CV_64F).var()))
        if cfg.fix_rotation:
            img, angle = _fix_rotation(img)
            turns += angle != 0
        if cfg.denoise:
            img = cv2.medianBlur(img, 3)
        if cfg.deskew:
            a = _estimate_skew(img)
            skews.append(abs(a))
            if abs(a) > 0.2:
                h, w = img.shape
                m = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
                img = cv2.warpAffine(img, m, (w, h), borderValue=255)
        pl, pc = _ocr_lines(img, pno)
        if cfg.fix_rotation and (not pc or np.mean(pc) < 60):
            # OSD misses upside down or sideways fax pages; retry the other orientations and keep the best read
            for rot in (cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
                tl, tc = _ocr_lines(cv2.rotate(img, rot), pno)
                if tc and np.mean(tc) * len(tc) > (np.mean(pc) * len(pc) if pc else 0) * 1.3:
                    pl, pc = tl, tc
                    turns += 1
                    break
        lines += pl
        confs += pc
        page_confs.append(float(np.mean(pc)) if pc else 0.0)
    n = len(doc)
    doc.close()
    words = sum(len(l.text.split()) for l in lines)
    feats = {
        "n_pages": n,
        "native_text_ratio": native / max(n, 1),
        "ocr_mean_conf": float(np.mean(confs)) if confs else 100.0,
        "ocr_low_conf_ratio": float(np.mean(np.array(confs) < 60)) if confs else 0.0,
        "skew_abs_mean": float(np.mean(skews)) if skews else 0.0,
        "rotated_pages": turns,
        "sharpness": float(np.mean(blur)) if blur else 0.0,
        "words_per_page": words / max(n, 1),
        "page_confs": page_confs,
        "ocr_seconds": time.perf_counter() - t0,
    }
    return PacketText(lines, feats)
