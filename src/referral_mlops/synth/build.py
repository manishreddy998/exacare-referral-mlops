"""Build a dataset of referral packets plus its answer key."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from referral_mlops.synth.degrade import degrade_pdf
from referral_mlops.synth.patients import make_cohort
from referral_mlops.synth.render import LAYOUTS, build_pages, draw_pdf

# share of packets per capture mode, by batch
MIXES = {
    "standard": {"digital": 0.45, "scan": 0.30, "fax": 0.25},
    "new_fax_source": {"digital": 0.10, "scan": 0.10, "bad_fax": 0.80},
}


def build_batch(out_dir: Path, n: int, seed: int, prefix: str, mix: str = "standard") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed + 7)
    modes, weights = zip(*MIXES[mix].items())
    manifest = out_dir / "manifest.jsonl"
    with manifest.open("w") as fh:
        for ref in make_cohort(n, seed, prefix):
            layout = rng.choice(LAYOUTS)
            mode = rng.choices(modes, weights)[0]
            clean = out_dir / f"{ref.packet_id}.digital.pdf"
            draw_pdf(build_pages(ref, rng, layout), clean, rng)
            path = clean
            if mode != "digital":
                path = out_dir / f"{ref.packet_id}.{mode}.pdf"
                degrade_pdf(clean, path, mode, rng)
                clean.unlink()
            fh.write(json.dumps({"packet_id": ref.packet_id, "path": path.name, "layout": layout,
                                 "capture": mode, "answer": ref.answer_key()}) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/packets")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build_batch(Path(args.out) / "main", args.n, args.seed, "PKT")
    build_batch(Path(args.out) / "drift", 80, args.seed + 1000, "NEW", mix="new_fax_source")


if __name__ == "__main__":
    main()
