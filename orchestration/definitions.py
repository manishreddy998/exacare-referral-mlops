"""Dagster assets for the referral pipeline. Each asset wraps one step in referral_mlops.pipeline."""

import json

from dagster import (AssetExecutionContext, Definitions, MaterializeResult, MetadataValue, ScheduleDefinition,
                     asset, define_asset_job)

from referral_mlops import pipeline as p


def _cfg():
    return p.load_cfg()


@asset(group_name="data", description="Synthetic referral packets (digital, scanned, faxed) with an answer key.")
def referral_packets(context: AssetExecutionContext) -> MaterializeResult:
    info = p.build_packets(_cfg())
    return MaterializeResult(metadata={"main_packets": info["main"], "drift_packets": info["drift"]})


def _eval_asset(model_name: str):
    @asset(name=f"eval_{model_name.replace('-', '_')}", deps=[referral_packets], group_name="models",
           description=f"OCR, extraction, calibration and test-split scoring for {model_name}; logged to MLflow.")
    def _a(context: AssetExecutionContext) -> MaterializeResult:
        r = p.evaluate_model(model_name, _cfg())
        return MaterializeResult(metadata={
            "micro_f1": round(r["metrics"]["micro_f1"], 4),
            "seconds_per_packet": round(r["cost"]["seconds_per_packet_mean"], 3),
            "auto_accept_share": round(r["review"]["auto_accept_share"], 4),
            "mlflow_version": r["mlflow"]["version"],
        })
    return _a


MODELS = ["rules-v1", "layout-v2", "layout-v2-fastocr"]
EVALS = [_eval_asset(m) for m in MODELS]


@asset(deps=EVALS, group_name="release", description="Release gate: promotes candidates that beat production.")
def release_gate(context: AssetExecutionContext) -> MaterializeResult:
    cfg = _cfg()
    evals = {m: json.loads((p.RESULTS / f"eval_{m}.json").read_text()) for m in MODELS}
    decisions = p.release(cfg, evals)
    table = "\n".join(f"| {d['candidate']} | {d['production']} | {'pass' if d['passed'] else 'BLOCK'} | "
                      f"{'; '.join(d['reasons'][:2]) or '-'} |" for d in decisions)
    return MaterializeResult(metadata={"decisions": MetadataValue.md(
        "| candidate | vs production | result | reasons |\n|---|---|---|---|\n" + table)})


@asset(deps=[release_gate], group_name="monitoring",
       description="Drift report, live accuracy and review queue for a new fax source, plus the feedback loop.")
def drift_monitor(context: AssetExecutionContext) -> MaterializeResult:
    prod = json.loads((p.RESULTS / "gate.json").read_text())["production_after"]
    out = p.monitor(_cfg(), prod)
    return MaterializeResult(metadata={"alerts": ", ".join(out["drift"]["alerts"]) or "none",
                                       "current_micro_f1": round(out["accuracy"]["current_micro_f1"], 4)})


@asset(deps=[drift_monitor], group_name="monitoring", description="Checks logs and MLflow for any patient value.")
def phi_audit(context: AssetExecutionContext) -> MaterializeResult:
    res = p.phi_audit()
    if not res["passed"]:
        raise RuntimeError(f"PHI found in {len(res['leaks'])} files")
    return MaterializeResult(metadata={"files_scanned": res["files_scanned"], "leaks": len(res["leaks"])})


@asset(deps=[phi_audit], group_name="monitoring", description="One summary file the dashboard and README read.")
def run_summary(context: AssetExecutionContext) -> MaterializeResult:
    evals = {m: json.loads((p.RESULTS / f"eval_{m}.json").read_text()) for m in MODELS}
    gate = json.loads((p.RESULTS / "gate.json").read_text())["decisions"]
    mon = json.loads((p.RESULTS / "drift.json").read_text())
    audit = json.loads((p.RESULTS / "phi_audit.json").read_text())
    s = p.summarize(evals, gate, mon, audit)
    return MaterializeResult(metadata={"production": json.loads((p.RESULTS / "gate.json").read_text())["production_after"]})


referral_job = define_asset_job("referral_pipeline", selection="*")

defs = Definitions(
    assets=[referral_packets, *EVALS, release_gate, drift_monitor, phi_audit, run_summary],
    jobs=[referral_job],
    schedules=[ScheduleDefinition(job=referral_job, cron_schedule="0 6 * * *", execution_timezone="America/Toronto")],
)
