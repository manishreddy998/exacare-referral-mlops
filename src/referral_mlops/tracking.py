"""MLflow helpers: experiment runs, a registered model per version, and the production alias."""
from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

REGISTERED = "referral-extractor"
EXPERIMENT = "referral-extraction"


def setup():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT)
    return MlflowClient()


class ReferralExtractor(mlflow.pyfunc.PythonModel):
    """Serves one model version: OCR, extraction and calibrated confidence, from a PDF path."""

    def load_context(self, context):
        import yaml

        from referral_mlops.models import ModelSpec
        self.spec = ModelSpec(**yaml.safe_load(Path(context.artifacts["spec"]).read_text()))
        cal = context.artifacts.get("calibrator")
        self.calibrator = pickle.loads(Path(cal).read_bytes()) if cal else None
        meta = json.loads(Path(context.artifacts["serving"]).read_text())
        self.threshold = meta["threshold"]
        self.unreadable_conf = meta["unreadable_conf"]
        self.min_ocr_conf = meta.get("min_ocr_conf", 0.0)

    def predict(self, context, model_input: pd.DataFrame, params=None):
        from referral_mlops.models import run_extractor
        from referral_mlops.ocr import read_packet
        from referral_mlops.review import route
        out = []
        for path in model_input["pdf_path"]:
            pt = read_packet(Path(path), self.spec.ocr_config)
            pred, secs = run_extractor(self.spec, pt, self.calibrator)
            res = route(pred, pt.features["page_confs"], self.threshold, self.unreadable_conf, self.min_ocr_conf)
            res["timing"] = {"ocr_seconds": pt.features["ocr_seconds"], "extract_seconds": secs}
            res["document"] = {k: v for k, v in pt.features.items() if k != "page_confs"}
            out.append(json.dumps(res))
        return out


def log_version(spec, spec_yaml: str, metrics: dict, params: dict, calibrator, serving: dict,
                extra_artifacts: dict[str, dict]) -> tuple[str, str]:
    """Logs one evaluation run and registers it. Returns (run_id, model_version)."""
    client = MlflowClient()
    with mlflow.start_run(run_name=spec.name) as run:
        mlflow.set_tags({"model_name": spec.name, "extractor": spec.extractor, "ocr": spec.ocr_key})
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "spec.yaml").write_text(spec_yaml)
            (d / "serving.json").write_text(json.dumps(serving))
            arts = {"spec": str(d / "spec.yaml"), "serving": str(d / "serving.json")}
            if calibrator is not None:
                (d / "calibrator.pkl").write_bytes(pickle.dumps(calibrator))
                arts["calibrator"] = str(d / "calibrator.pkl")
            for name, obj in extra_artifacts.items():
                (d / name).write_text(json.dumps(obj, indent=2))
                mlflow.log_artifact(str(d / name), "reports")
            src = Path(__file__).resolve().parent
            mlflow.pyfunc.log_model(name="model", python_model=ReferralExtractor(), artifacts=arts,
                                    code_paths=[str(src)], registered_model_name=REGISTERED)
        versions = client.search_model_versions(f"name='{REGISTERED}' and run_id='{run.info.run_id}'")
        version = versions[0].version
        client.set_model_version_tag(REGISTERED, version, "model_name", spec.name)
    return run.info.run_id, version


def production_version(client: MlflowClient):
    try:
        return client.get_model_version_by_alias(REGISTERED, "production")
    except Exception:
        return None


def promote(client: MlflowClient, version: str):
    client.set_registered_model_alias(REGISTERED, "production", version)


def tracking_dump(client: MlflowClient) -> dict:
    """Everything MLflow stores as text, for the PHI audit."""
    exp = client.get_experiment_by_name(EXPERIMENT)
    runs = client.search_runs([exp.experiment_id]) if exp else []
    return {"runs": [{"params": r.data.params, "tags": r.data.tags, "metrics": list(r.data.metrics)} for r in runs],
            "versions": [{"version": v.version, "tags": v.tags, "aliases": v.aliases}
                         for v in client.search_model_versions(f"name='{REGISTERED}'")]}
