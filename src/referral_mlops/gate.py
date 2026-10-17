"""CI release gate: score production and candidate on a fresh synthetic batch and block a worse candidate.

Exit code 0 means the candidate may ship, 1 means it is blocked.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import yaml

from referral_mlops import pipeline as p


def run(production: str, candidate: str, packets: int, seed: int) -> dict:
    cfg = p.load_cfg()
    cfg["data"] |= {"main_packets": packets, "drift_packets": 4, "seed": seed}
    cfg["workers"] = int(os.environ.get("GATE_WORKERS", os.cpu_count() or 2))
    with tempfile.TemporaryDirectory() as tmp:
        p.DATA, p.RESULTS = Path(tmp) / "data", Path(tmp) / "results"
        p.build_packets(cfg)
        evals = {m: p.evaluate_model(m, cfg, track=False) for m in {production, candidate}}
    return p.gate_decision(evals[production], evals[candidate], cfg["gate"])


def main():
    ap = argparse.ArgumentParser()
    rel = yaml.safe_load((p.ROOT / "configs" / "release.yaml").read_text())
    ap.add_argument("--production", default=rel["production"])
    ap.add_argument("--candidate", default=rel["candidate"])
    ap.add_argument("--packets", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--expect-block", action="store_true", help="negative control: succeed only if blocked")
    a = ap.parse_args()
    dec = run(a.production, a.candidate, a.packets, a.seed)
    lines = [f"### Release gate: {dec['candidate']} vs {dec['production']}",
             f"Result: **{'PASS' if dec['passed'] else 'BLOCKED'}**",
             f"Micro F1: {dec['micro_f1']['production']:.4f} (production) vs {dec['micro_f1']['candidate']:.4f}",
             f"Seconds per packet: {dec['seconds_per_packet']['production']:.2f} vs "
             f"{dec['seconds_per_packet']['candidate']:.2f}"] + [f"- {r}" for r in dec["reasons"]]
    print("\n".join(lines))
    print(json.dumps(dec))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
            fh.write("\n".join(lines) + "\n")
    ok = (not dec["passed"]) if a.expect_block else dec["passed"]
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
