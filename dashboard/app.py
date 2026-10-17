"""Operations dashboard for the referral extraction pipeline."""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

R = Path(__file__).resolve().parents[1] / "results"
st.set_page_config(page_title="Referral MLOps", layout="wide")


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


summary, gate, drift = load("summary.json"), load("gate.json"), load("drift.json")
if not summary:
    st.warning("No results yet. Run the pipeline first: docker compose up --build")
    st.stop()

models = list(summary["models"])
evals = {m: load(f"eval_{m}.json") for m in models}
prod = gate["production_after"]
pe = evals[prod]

st.title("Referral packet extraction: model operations")
st.caption("Synthetic referral packets only. No real patient data.")

c = st.columns(6)
c[0].metric("Production model", prod, f"registry v{pe['mlflow']['version']}")
c[1].metric("Test micro F1", f"{pe['metrics']['micro_f1']:.3f}",
            f"{pe['metrics']['micro_f1'] - evals[models[0]]['metrics']['micro_f1']:+.3f} vs {models[0]}")
c[2].metric("Fields auto-accepted", f"{pe['review']['auto_accept_share']:.1%}",
            f"precision {pe['review']['auto_accept_precision']:.2%}", delta_color="off")
c[3].metric("CPU seconds / packet", f"{pe['cost']['seconds_per_packet_mean']:.2f}",
            f"p95 {pe['cost']['seconds_per_packet_p95']:.1f}s", delta_color="off")
c[4].metric("Compute $ / 1,000 packets", f"${pe['cost']['usd_per_1000_packets']:.3f}", "assumed rate",
            delta_color="off")
c[5].metric("PHI audit", "passed" if summary["phi_audit_passed"] else "FAILED",
            f"{load('phi_audit.json')['files_scanned']} files", delta_color="off")

st.subheader("Model versions on the held-out test split")
rows = []
for m, e in evals.items():
    rows.append({"model": m, "registry version": e["mlflow"]["version"], "micro F1": e["metrics"]["micro_f1"],
                 "precision": e["metrics"]["micro_precision"], "recall": e["metrics"]["micro_recall"],
                 "packets fully correct": e["metrics"]["packet_exact_match"],
                 "sec / packet": e["cost"]["seconds_per_packet_mean"], "auto-accepted": e["review"]["auto_accept_share"]})
st.dataframe(pd.DataFrame(rows).style.format({k: "{:.3f}" for k in rows[0] if k not in ("model", "registry version")}),
             hide_index=True, use_container_width=True)

left, right = st.columns(2)
fld = pd.DataFrame([{"model": m, "field": f, "F1": v["f1"]} for m, e in evals.items()
                    for f, v in e["metrics"]["fields"].items()])
left.plotly_chart(px.bar(fld, x="field", y="F1", color="model", barmode="group", range_y=[0, 1.02],
                         title="Field level F1"), use_container_width=True)
cap = pd.DataFrame([{"model": m, "capture": k, "micro F1": v["micro_f1"]} for m, e in evals.items()
                    for k, v in e["by_capture"].items()])
right.plotly_chart(px.bar(cap, x="capture", y="micro F1", color="model", barmode="group", range_y=[0, 1.02],
                          title="Accuracy by how the packet arrived"), use_container_width=True)

st.subheader("Release gate")
for d in gate["decisions"]:
    icon = "✅ promoted" if d["passed"] else "⛔ blocked"
    st.markdown(f"**{d['candidate']}** vs production **{d['production']}**: {icon}. "
                f"Micro F1 {d['micro_f1']['production']:.4f} to {d['micro_f1']['candidate']:.4f}, "
                f"{d['seconds_per_packet']['production']:.2f}s to {d['seconds_per_packet']['candidate']:.2f}s per packet.")
    if d["reasons"]:
        st.caption("Reasons: " + "; ".join(d["reasons"][:4]) + (" ..." if len(d["reasons"]) > 4 else ""))

st.subheader("Drift: new fax source batch vs reference")
l2, r2 = st.columns([3, 2])
feat = pd.DataFrame([{"feature": f, "PSI": v["psi"], "control PSI": drift["control_check"]["features"][f]["psi"]}
                     for f, v in drift["drift"]["features"].items()])
fig = go.Figure([go.Bar(name="new source vs reference", x=feat["feature"], y=feat["PSI"]),
                 go.Bar(name="control (train vs test)", x=feat["feature"], y=feat["control PSI"])])
fig.add_hline(y=0.2, line_dash="dash", annotation_text="alert at 0.2")
fig.update_layout(barmode="group", title="Population stability index", yaxis_type="log")
l2.plotly_chart(fig, use_container_width=True)
acc = drift["accuracy"]
r2.metric("Micro F1 on reference", f"{acc['reference_micro_f1']:.3f}")
r2.metric("Micro F1 on new fax source", f"{acc['current_micro_f1']:.3f}",
          f"{acc['current_micro_f1'] - acc['reference_micro_f1']:+.3f}")
fb = drift["feedback_loop"]
r2.markdown(f"**Reviewer feedback loop** ({fb['reviewed_packets']} reviewed packets retrain the calibrator, "
            f"{fb['held_out_packets']} held out):")
r2.dataframe(pd.DataFrame({
    "": ["errors sent to review", "auto-accept precision", "share sent to review"],
    "before": [f"{fb['before']['wrong_values_caught_by_review']:.0%}", f"{fb['before']['auto_accept_precision']:.2%}",
               f"{fb['before']['review_share']:.0%}"],
    "after": [f"{fb['after']['wrong_values_caught_by_review']:.0%}", f"{fb['after']['auto_accept_precision']:.2%}",
              f"{fb['after']['review_share']:.0%}"]}), hide_index=True, use_container_width=True)

st.subheader("Cost and latency per packet")
pp = pd.DataFrame(pe["per_packet"])
l3, r3 = st.columns(2)
l3.plotly_chart(px.box(pp, x="capture", y="seconds", points="all", title=f"{prod}: CPU seconds per packet"),
                use_container_width=True)
cost = pd.DataFrame([{"model": m, "sec / packet": e["cost"]["seconds_per_packet_mean"],
                      "micro F1": e["metrics"]["micro_f1"]} for m, e in evals.items()])
r3.plotly_chart(px.scatter(cost, x="sec / packet", y="micro F1", text="model", title="Accuracy vs cost",
                           range_y=[0.7, 1.01]).update_traces(textposition="top center", marker_size=14),
                use_container_width=True)

st.subheader("Human review queue (new fax source)")
q = pd.read_json(R / "review_queue.jsonl", lines=True) if (R / "review_queue.jsonl").exists() else pd.DataFrame()
if len(q):
    s1, s2 = st.columns([1, 3])
    s1.dataframe(q.groupby("reason").size().rename("fields").reset_index(), hide_index=True)
    s1.dataframe(q.groupby("field").size().rename("fields").sort_values(ascending=False).reset_index(), hide_index=True)
    s2.dataframe(q.sort_values("confidence").head(200), hide_index=True, use_container_width=True, height=380)
