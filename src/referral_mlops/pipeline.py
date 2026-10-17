"""Pipeline steps. Dagster assets call these; `python -m referral_mlops.pipeline run` runs them in order."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from referral_mlops import calibrate, drift, phi, review
from referral_mlops.evaluate import score
from referral_mlops.extract.common import ALL_FIELDS
from referral_mlops.models import ModelSpec, run_extractor
from referral_mlops.ocr import OcrConfig, read_packet
from referral_mlops.synth.build import build_batch

ROOT = Path(__file__).resolve().parents[2]
DATA, RESULTS = ROOT / "data", ROOT / "results"
log = phi.get_logger("pipeline", RESULTS / "logs" / "pipeline.log")


def load_cfg() -> dict:
    path = os.environ.get("REFERRAL_CONFIG", str(ROOT / "configs" / "pipeline.yaml"))
    return yaml.safe_load(Path(path).read_text())


def manifest(batch: str) -> list[dict]:
    return [json.loads(l) for l in (DATA / "packets" / batch / "manifest.jsonl").open()]


def build_packets(cfg: dict) -> dict:
    d = cfg["data"]
    stamp = DATA / "packets" / "build.json"
    want = {"main": d["main_packets"], "drift": d["drift_packets"], "seed": d["seed"]}
    if stamp.exists() and json.loads(stamp.read_text()) == want:
        log.info("packets already built %s", want)
        return want
    t0 = time.perf_counter()
    build_batch(DATA / "packets" / "main", d["main_packets"], d["seed"], "PKT")
    build_batch(DATA / "packets" / "drift", d["drift_packets"], d["seed"] + 1000, "NEW", mix="new_fax_source")
    stamp.write_text(json.dumps(want))
    log.info("built packets in %.1fs", time.perf_counter() - t0)
    return want


def _ocr_one(args):
    path, cfg = args
    return read_packet(path, OcrConfig(**cfg))


def ocr_batch(batch: str, spec: ModelSpec, workers: int) -> list:
    cache = DATA / "ocr" / spec.ocr_key / f"{batch}.pkl"
    rows = manifest(batch)
    sig = hashlib.sha1(json.dumps([r["packet_id"] + r["path"] for r in rows]).encode()).hexdigest()
    if cache.exists():
        saved = pickle.loads(cache.read_bytes())
        if saved["sig"] == sig:
            return saved["texts"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    jobs = [(DATA / "packets" / batch / r["path"], spec.ocr) for r in rows]
    with ProcessPoolExecutor(workers) as ex:
        texts = list(ex.map(_ocr_one, jobs, chunksize=2))
    cache.write_bytes(pickle.dumps({"sig": sig, "texts": texts}))
    log.info("ocr %s with %s: %d packets in %.1fs wall", batch, spec.ocr_key, len(texts), time.perf_counter() - t0)
    return texts


def _split(cfg, n):
    cut = int(n * cfg["data"]["train_share"])
    return slice(0, cut), slice(cut, n)


def _doc_row(pt, pred, threshold, min_ocr):
    res = [r for rs in pred.values() for r in rs]
    feats = {k: v for k, v in pt.features.items() if k != "page_confs"}
    feats["mean_field_confidence"] = float(np.mean([r.confidence for r in res])) if res else 0.0
    feats["review_share"] = (float(np.mean([review.needs_review(r.confidence, r.features, threshold, min_ocr)
                                            for r in res])) if res else 1.0)
    return feats


def evaluate_model(name: str, cfg: dict) -> dict:
    """Train the calibrator on the train split, then score everything on the held out test split."""
    spec = ModelSpec.load(name)
    rows = manifest("main")
    texts = ocr_batch("main", spec, cfg["workers"])
    golds = [r["answer"] for r in rows]
    tr, te = _split(cfg, len(rows))
    raw = [run_extractor(spec, pt)[0] for pt in texts[tr]]
    calibrator, cal_stats = None, {}
    if spec.calibrate:
        calibrator, cal_stats = calibrate.fit(raw, golds[tr], texts[tr])
        raw = [run_extractor(spec, pt, calibrator)[0] for pt in texts[tr]]
    threshold = review.choose_threshold(raw, golds[tr], cfg["review"]["target_error_capture"],
                                        cfg["review"]["max_review_share"])
    min_ocr = cfg["review"]["min_ocr_conf"]
    train_rows = [_doc_row(pt, p, threshold, min_ocr) for pt, p in zip(texts[tr], raw)]

    preds, ext_secs = [], []
    for pt in texts[te]:
        p, s = run_extractor(spec, pt, calibrator)
        preds.append(p)
        ext_secs.append(s)
    m = score(preds, golds[te])
    ocr_secs = [pt.features["ocr_seconds"] for pt in texts[te]]
    total = np.array(ocr_secs) + np.array(ext_secs)
    rate = cfg["cost"]["usd_per_cpu_second"]
    by_capture = {}
    for cap in ("digital", "scan", "fax"):
        idx = [i for i, r in enumerate(rows[te]) if r["capture"] == cap]
        if idx:
            s = score([preds[i] for i in idx], [golds[te][i] for i in idx])
            by_capture[cap] = {"packets": len(idx), "micro_f1": s["micro_f1"],
                               "seconds_per_packet": float(np.mean(total[idx]))}
    rv = review.review_stats(preds, golds[te], texts[te], threshold, cfg["review"]["unreadable_page_conf"],
                             cfg["review"]["min_ocr_conf"])
    result = {
        "model": name, "description": spec.description, "test_packets": len(preds),
        "metrics": m, "by_capture": by_capture, "review": rv, "calibrator": cal_stats,
        "cost": {"seconds_per_packet_mean": float(total.mean()), "seconds_per_packet_p50": float(np.median(total)),
                 "seconds_per_packet_p95": float(np.percentile(total, 95)),
                 "ocr_seconds_mean": float(np.mean(ocr_secs)), "extract_ms_mean": float(np.mean(ext_secs) * 1000),
                 "usd_per_packet": float(total.mean() * rate), "usd_per_1000_packets": float(total.mean() * rate * 1000)},
        "serving": {"threshold": threshold, "unreadable_conf": cfg["review"]["unreadable_page_conf"],
                    "min_ocr_conf": cfg["review"]["min_ocr_conf"]},
        "train_per_packet": train_rows,
        "per_packet": [{"packet_id": r["packet_id"], "capture": r["capture"], "layout": r["layout"],
                        "seconds": float(t), **_doc_row(pt, p, threshold, min_ocr)}
                       for r, pt, p, t in zip(rows[te], texts[te], preds, total)],
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"eval_{name}.json").write_text(json.dumps(result, indent=2))
    log.info("evaluated %s micro_f1=%.4f cost=%.2fs/packet", name, m["micro_f1"], result["cost"]["seconds_per_packet_mean"])

    from referral_mlops import tracking
    tracking.setup()
    flat = {"micro_f1": m["micro_f1"], "micro_precision": m["micro_precision"], "micro_recall": m["micro_recall"],
            "packet_exact_match": m["packet_exact_match"], "min_field_f1": m["min_field_f1"],
            "seconds_per_packet": result["cost"]["seconds_per_packet_mean"],
            "seconds_per_packet_p95": result["cost"]["seconds_per_packet_p95"],
            "usd_per_1000_packets": result["cost"]["usd_per_1000_packets"],
            "auto_accept_share": rv["auto_accept_share"], "auto_accept_precision": rv["auto_accept_precision"],
            "review_threshold": threshold}
    flat |= {f"f1_{f}": m["fields"][f]["f1"] for f in ALL_FIELDS}
    flat |= {f"precision_{f}": m["fields"][f]["precision"] for f in ALL_FIELDS}
    flat |= {f"recall_{f}": m["fields"][f]["recall"] for f in ALL_FIELDS}
    flat |= {f"calibrator_{k}": v for k, v in cal_stats.items()}
    params = {"extractor": spec.extractor, **{f"ocr_{k}": v for k, v in spec.ocr.items()},
              "calibrate": spec.calibrate, "test_packets": len(preds), "seed": cfg["data"]["seed"],
              "dataset_sha": hashlib.sha1(json.dumps([r["packet_id"] for r in rows]).encode()).hexdigest()[:12]}
    report = {k: v for k, v in result.items() if k not in ("per_packet", "train_per_packet")}
    run_id, version = tracking.log_version(spec, (ROOT / "configs" / "models" / f"{name}.yaml").read_text(), flat,
                                           params, calibrator, result["serving"], {"evaluation.json": report})
    result["mlflow"] = {"run_id": run_id, "version": version}
    (RESULTS / f"eval_{name}.json").write_text(json.dumps(result, indent=2))
    return result


def gate_decision(prod: dict, cand: dict, g: dict) -> dict:
    pm, cm = prod["metrics"], cand["metrics"]
    reasons = []
    d = cm["micro_f1"] - pm["micro_f1"]
    if d < -g["max_micro_f1_drop"]:
        reasons.append(f"micro F1 fell {pm['micro_f1']:.4f} to {cm['micro_f1']:.4f}")
    for f in ALL_FIELDS:
        fd = cm["fields"][f]["f1"] - pm["fields"][f]["f1"]
        if fd < -g["max_field_f1_drop"]:
            reasons.append(f"{f} F1 fell {pm['fields'][f]['f1']:.3f} to {cm['fields'][f]['f1']:.3f}")
    pc, cc = prod["cost"]["seconds_per_packet_mean"], cand["cost"]["seconds_per_packet_mean"]
    if cc > pc * (1 + g["max_cost_increase"]) and d < g["min_gain_for_cost"]:
        reasons.append(f"cost per packet rose {pc:.2f}s to {cc:.2f}s without a {g['min_gain_for_cost']:.0%} F1 gain")
    p95 = cand["cost"]["seconds_per_packet_p95"]
    if p95 > g["max_p95_seconds"]:
        reasons.append(f"p95 latency {p95:.1f}s is over the {g['max_p95_seconds']}s budget")
    return {"production": prod["model"], "candidate": cand["model"], "passed": not reasons, "reasons": reasons,
            "micro_f1": {"production": pm["micro_f1"], "candidate": cm["micro_f1"]},
            "seconds_per_packet": {"production": pc, "candidate": cc}}


def release(cfg: dict, evals: dict) -> list[dict]:
    """Bootstraps production, then runs each candidate through the gate in order and promotes winners."""
    from referral_mlops import tracking
    client = tracking.setup()
    current = cfg["release"]["production"]
    tracking.promote(client, evals[current]["mlflow"]["version"])
    decisions = []
    for cand in cfg["release"]["candidates"]:
        dec = gate_decision(evals[current], evals[cand], cfg["gate"])
        dec["candidate_version"] = evals[cand]["mlflow"]["version"]
        if dec["passed"]:
            tracking.promote(client, dec["candidate_version"])
            current = cand
        client.set_model_version_tag(tracking.REGISTERED, dec["candidate_version"], "gate",
                                     "passed" if dec["passed"] else "blocked")
        decisions.append(dec)
        log.info("gate %s vs %s: %s %s", cand, dec["production"], "PASS" if dec["passed"] else "BLOCK", dec["reasons"])
    out = {"decisions": decisions, "production_after": current}
    (RESULTS / "gate.json").write_text(json.dumps(out, indent=2))
    return decisions


def monitor(cfg: dict, production: str) -> dict:
    """Drift and live accuracy of the production model on a batch from a new, worse fax source."""
    ev = json.loads((RESULTS / f"eval_{production}.json").read_text())
    spec = ModelSpec.load(production)
    import mlflow
    from referral_mlops import tracking
    tracking.setup()
    model = mlflow.pyfunc.load_model(f"models:/{tracking.REGISTERED}@production")
    calibrator = model.unwrap_python_model().calibrator
    thr = ev["serving"]["threshold"]
    rows = manifest("drift")
    texts = ocr_batch("drift", spec, cfg["workers"])
    preds = [run_extractor(spec, pt, calibrator)[0] for pt in texts]
    cur = [_doc_row(pt, p, thr, cfg["review"]["min_ocr_conf"]) for pt, p in zip(texts, preds)]
    ref = ev["per_packet"]
    # control: training packets against test packets from the same source should raise no alert
    control = drift.report(ev["train_per_packet"], ref, cfg["drift"]["psi_alert"])
    rep = drift.report(ref, cur, cfg["drift"]["psi_alert"])
    s = score(preds, [r["answer"] for r in rows])
    rc = cfg["review"]
    rv = review.review_stats(preds, [r["answer"] for r in rows], texts, thr, rc["unreadable_page_conf"],
                             rc["min_ocr_conf"])
    out = {"production": production, "reference_packets": len(ref), "current_packets": len(cur),
           "control_check": control, "drift": rep,
           "accuracy": {"reference_micro_f1": ev["metrics"]["micro_f1"], "current_micro_f1": s["micro_f1"],
                        "current_fields": {f: v["f1"] for f, v in s["fields"].items()}},
           "review": rv, "per_packet": [{"packet_id": r["packet_id"], "capture": r["capture"], **c}
                                        for r, c in zip(rows, cur)]}
    (RESULTS / "drift.json").write_text(json.dumps(out, indent=2))
    queue = []
    for r, pt, p in zip(rows, texts, preds):
        routed = review.route(p, pt.features["page_confs"], thr, rc["unreadable_page_conf"], rc["min_ocr_conf"])
        for f, items in routed["fields"].items():
            for it in items:
                if it["decision"] == "review" or routed["packet_review"]:
                    queue.append({"packet_id": r["packet_id"], "field": f, **it,
                                  "reason": "unreadable page" if routed["packet_review"] else "low confidence"})
    with (RESULTS / "review_queue.jsonl").open("w") as fh:
        for q in queue:
            fh.write(json.dumps(q) + "\n")
    log.info("drift alerts %s, current micro_f1=%.4f, review items=%d", rep["alerts"], s["micro_f1"], len(queue))
    out["feedback_loop"] = feedback_loop(cfg, spec, rows, texts, calibrator, thr)
    (RESULTS / "drift.json").write_text(json.dumps(out, indent=2))
    return out


def feedback_loop(cfg, spec, rows, texts, old_cal, old_thr) -> dict:
    """Reviewers correct the first half of the new source batch. Their labels retrain the calibrator, and the
    second half (not seen in training) shows whether routing got better at catching errors."""
    rc = cfg["review"]
    half = len(rows) // 2
    main_rows = manifest("main")
    main_texts = ocr_batch("main", spec, cfg["workers"])
    tr, _ = _split(cfg, len(main_rows))
    base_raw = [run_extractor(spec, pt)[0] for pt in main_texts[tr]]
    fb_raw = [run_extractor(spec, pt)[0] for pt in texts[:half]]
    golds = [r["answer"] for r in main_rows[tr]] + [r["answer"] for r in rows[:half]]
    new_cal, stats = calibrate.fit(base_raw + fb_raw, golds, list(main_texts[tr]) + list(texts[:half]))
    tuned = [run_extractor(spec, pt, new_cal)[0] for pt in list(main_texts[tr]) + list(texts[:half])]
    new_thr = review.choose_threshold(tuned, golds, rc["target_error_capture"], rc["max_review_share"])
    held_gold = [r["answer"] for r in rows[half:]]
    before = review.review_stats([run_extractor(spec, pt, old_cal)[0] for pt in texts[half:]], held_gold,
                                 texts[half:], old_thr, rc["unreadable_page_conf"], rc["min_ocr_conf"])
    after = review.review_stats([run_extractor(spec, pt, new_cal)[0] for pt in texts[half:]], held_gold,
                                texts[half:], new_thr, rc["unreadable_page_conf"], rc["min_ocr_conf"])
    res = {"reviewed_packets": half, "held_out_packets": len(rows) - half, "calibrator": stats,
           "before": before, "after": after}
    from referral_mlops import tracking
    import mlflow
    tracking.setup()
    with mlflow.start_run(run_name=f"{spec.name}-recalibrated"):
        mlflow.set_tags({"model_name": f"{spec.name}-recalibrated", "kind": "feedback_loop"})
        mlflow.log_params({"reviewed_packets": half, "held_out_packets": len(rows) - half})
        mlflow.log_metrics({f"before_{k}": v for k, v in before.items() if isinstance(v, (int, float))} |
                           {f"after_{k}": v for k, v in after.items() if isinstance(v, (int, float))})
    log.info("feedback loop: errors caught %.2f -> %.2f, auto precision %.4f -> %.4f",
             before["wrong_values_caught_by_review"], after["wrong_values_caught_by_review"],
             before["auto_accept_precision"], after["auto_accept_precision"])
    return res


def phi_audit() -> dict:
    from referral_mlops import tracking
    client = tracking.setup()
    dump = RESULTS / "logs" / "mlflow_tracking_dump.json"
    dump.write_text(json.dumps(tracking.tracking_dump(client), indent=2))
    keys = manifest("main") + manifest("drift")
    targets = [RESULTS / "logs"] + [p for p in (ROOT / "mlartifacts", ROOT / "mlruns") if p.exists()]
    res = phi.audit(targets, [k["answer"] for k in keys])
    (RESULTS / "phi_audit.json").write_text(json.dumps(res, indent=2))
    log.info("phi audit scanned %d files, leaks=%d", res["files_scanned"], len(res["leaks"]))
    return res


def summarize(evals: dict, gate: list, mon: dict, audit: dict) -> dict:
    s = {"models": {n: {"micro_f1": e["metrics"]["micro_f1"], "micro_precision": e["metrics"]["micro_precision"],
                        "micro_recall": e["metrics"]["micro_recall"],
                        "packet_exact_match": e["metrics"]["packet_exact_match"],
                        "seconds_per_packet": e["cost"]["seconds_per_packet_mean"],
                        "usd_per_1000_packets": e["cost"]["usd_per_1000_packets"],
                        "auto_accept_share": e["review"]["auto_accept_share"],
                        "auto_accept_precision": e["review"]["auto_accept_precision"],
                        "by_capture": e["by_capture"], "version": e["mlflow"]["version"]} for n, e in evals.items()},
         "gate": [{k: d[k] for k in ("candidate", "production", "passed", "reasons")} for d in gate],
         "drift_alerts": mon["drift"]["alerts"], "control_alerts": mon["control_check"]["alerts"],
         "drift_accuracy": mon["accuracy"], "drift_review": mon["review"], "phi_audit_passed": audit["passed"]}
    (RESULTS / "summary.json").write_text(json.dumps(s, indent=2))
    return s


def run_all():
    cfg = load_cfg()
    build_packets(cfg)
    names = [cfg["release"]["production"]] + cfg["release"]["candidates"]
    evals = {n: evaluate_model(n, cfg) for n in names}
    decisions = release(cfg, evals)
    prod = json.loads((RESULTS / "gate.json").read_text())["production_after"]
    mon = monitor(cfg, prod)
    audit = phi_audit()
    return summarize(evals, decisions, mon, audit)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "build", "audit"])
    a = ap.parse_args()
    if a.cmd == "run":
        print(json.dumps(run_all(), indent=2))
    elif a.cmd == "build":
        build_packets(load_cfg())
    else:
        r = phi_audit()
        raise SystemExit(0 if r["passed"] else 1)
