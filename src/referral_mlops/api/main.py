"""Serves the production model from the MLflow registry."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

from referral_mlops.phi import get_logger

log = get_logger("api", Path(os.environ.get("API_LOG", "results/logs/api.log")))
app = FastAPI(title="Referral packet extraction", version="1.0")
_state: dict = {}


def _model():
    if "model" not in _state:
        import mlflow

        from referral_mlops.tracking import REGISTERED, setup
        client = setup()
        mv = client.get_model_version_by_alias(REGISTERED, "production")
        _state["model"] = mlflow.pyfunc.load_model(f"models:/{REGISTERED}@production")
        _state["version"] = mv.version
        _state["name"] = mv.tags.get("model_name", "")
    return _state["model"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model")
def model_info():
    _model()
    return {"registered_model": "referral-extractor", "alias": "production", "version": _state["version"],
            "model_name": _state["name"]}


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    body = await file.read()
    if not body.startswith(b"%PDF"):
        raise HTTPException(400, "expected a PDF")
    doc_id = hashlib.sha256(body).hexdigest()[:12]
    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(body)
        tmp.flush()
        res = json.loads(_model().predict(pd.DataFrame({"pdf_path": [tmp.name]}))[0])
    ms = (time.perf_counter() - t0) * 1000
    n_review = sum(1 for items in res["fields"].values() for i in items if i["decision"] == "review")
    # only identifiers that carry no patient data are logged
    log.info("doc=%s model_version=%s pages=%d latency_ms=%.0f review_fields=%d packet_review=%s",
             doc_id, _state["version"], res["document"]["n_pages"], ms, n_review, res["packet_review"])
    res["doc_id"] = doc_id
    res["model_version"] = _state["version"]
    res["latency_ms"] = round(ms, 1)
    return res
