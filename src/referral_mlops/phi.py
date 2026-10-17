"""Keep patient data out of logs, and prove it with an audit against the answer key."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

PATTERNS = [
    (re.compile(r"\b[A-Z]\d{9}\b"), "[MEMBER_ID]"),
    (re.compile(r"\bMRN\d+\b"), "[MRN]"),
    (re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b|\b\d{2}-[A-Z]{3}-\d{4}\b"), "[DATE]"),
    (re.compile(r"\(\d{3}\)\s?\d{3}-\d{4}"), "[PHONE]"),
    (re.compile(r"(?i)\b(name|patient|pt|dob|member id|policy #?|subscriber id)\s*[:=]\s*[^,;}\n]+"), r"\1: [REDACTED]"),
]


def redact(text: str) -> str:
    for rx, rep in PATTERNS:
        text = rx.sub(rep, text)
    return text


class PhiFilter(logging.Filter):
    """Redacts identifiers in every record before any handler writes it."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = ()
        return True


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    log = logging.getLogger(name)
    if getattr(log, "_phi_ready", False):
        return log
    log.setLevel(logging.INFO)
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handlers = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    for h in handlers:
        h.setFormatter(fmt)
        h.addFilter(PhiFilter())
        log.addHandler(h)
    log._phi_ready = True
    return log


def sensitive_values(answer_keys: list[dict]) -> set[str]:
    vals = set()
    for a in answer_keys:
        vals.add(a["patient_name"].lower())
        vals.add(a["patient_name"].split()[-1].lower() + ", " + a["patient_name"].split()[0].lower())
        vals.add(a["dob"])
        vals.add(a["member_id"].lower())
    return {v for v in vals if len(v) >= 6}


def audit(paths: list[Path], answer_keys: list[dict]) -> dict:
    """Search every log and tracking file for any patient value from the answer key."""
    needles = sensitive_values(answer_keys)
    hits, scanned = [], 0
    for p in paths:
        files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
        for f in files:
            if f.suffix in {".pkl", ".pdf", ".png", ".db"} or f.stat().st_size > 20_000_000:
                continue
            scanned += 1
            text = f.read_text(errors="ignore").lower()
            for n in needles:
                if n in text:
                    hits.append({"file": str(f), "kind": "answer key value"})
                    break
    return {"files_scanned": scanned, "values_checked": len(needles), "leaks": hits, "passed": not hits}


if __name__ == "__main__":
    import sys
    keys = [json.loads(l)["answer"] for m in Path("data/packets").rglob("manifest.jsonl") for l in m.open()]
    res = audit([Path(p) for p in sys.argv[1:]], keys)
    print(json.dumps({k: v for k, v in res.items() if k != "leaks"} | {"leak_count": len(res["leaks"])}))
    sys.exit(0 if res["passed"] else 1)
