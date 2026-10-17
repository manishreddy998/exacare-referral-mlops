"""End to end: register a model in a throwaway MLflow store, serve it, and check the logs hold no PHI."""
import json
import logging
from pathlib import Path

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch, packet_dir):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("API_LOG", str(tmp_path / "api.log"))
    from referral_mlops import tracking
    from referral_mlops.calibrate import HeuristicCalibrator
    from referral_mlops.models import CONFIG_DIR, ModelSpec
    client = tracking.setup()
    spec = ModelSpec.load("layout-v2")
    _, version = tracking.log_version(spec, (CONFIG_DIR / "layout-v2.yaml").read_text(), {"micro_f1": 1.0},
                                      {"test": 1}, HeuristicCalibrator(),
                                      {"threshold": 0.5, "unreadable_conf": 50, "min_ocr_conf": 0.8}, {})
    tracking.promote(client, version)
    import importlib

    from referral_mlops.api import main
    importlib.reload(main)
    from fastapi.testclient import TestClient
    return TestClient(main.app), tmp_path


def test_extract_serves_production_and_logs_no_phi(client, packet_dir):
    c, tmp = client
    row = next(json.loads(l) for l in (packet_dir / "manifest.jsonl").open() if "digital" in l)
    r = c.post("/extract", files={"file": ("p.pdf", (packet_dir / row["path"]).read_bytes(), "application/pdf")})
    assert r.status_code == 200
    body = r.json()
    assert body["fields"]["patient_name"][0]["value"] == row["answer"]["patient_name"].lower()
    assert c.get("/model").json()["alias"] == "production"
    for h in logging.getLogger("api").handlers:
        h.flush()
    text = (tmp / "api.log").read_text().lower()
    assert "latency_ms" in text
    assert row["answer"]["patient_name"].lower() not in text
    assert row["answer"]["member_id"].lower() not in text


def test_rejects_non_pdf(client):
    c, _ = client
    assert c.post("/extract", files={"file": ("x.txt", b"hello", "text/plain")}).status_code == 400
