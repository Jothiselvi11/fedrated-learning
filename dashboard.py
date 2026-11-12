"""
FedGuard-IoT security dashboard (demo).

Shows training results and lets you "capture" a random network flow, run it through
the federated global model, and see the alert plus the features that caused it.

Run `python fedguard.py` first, then:  python dashboard.py  ->  http://127.0.0.1:5000
"""
import json
import os

import numpy as np
from flask import Flask, jsonify, send_file

import fedguard as F

HERE = os.path.dirname(os.path.abspath(__file__))
data = np.load(os.path.join(HERE, "global_model.npz"))
MODEL = {k: data[k] for k in ("W1", "b1", "W2", "b2")}
XTE, YTE, BASELINE = data["Xte"], data["yte"], data["baseline"]
with open(os.path.join(HERE, "results.json")) as f:
    RESULTS = json.load(f)

app = Flask(__name__)
rng = np.random.default_rng()

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>FedGuard-IoT Dashboard</title>
<style>
 body{font-family:Segoe UI,Arial,sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:24px}
 h1{margin:0 0 4px} .sub{color:#94a3b8;margin-bottom:20px}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
 .card{background:#1e293b;border-radius:12px;padding:18px}
 table{width:100%;border-collapse:collapse} td,th{padding:6px 8px;border-bottom:1px solid #334155;text-align:left}
 button{background:#38bdf8;border:0;color:#0f172a;font-weight:600;padding:10px 18px;border-radius:8px;cursor:pointer}
 .alert{color:#f87171;font-size:22px;font-weight:700} .ok{color:#4ade80;font-size:22px;font-weight:700}
 .bar{height:12px;background:#f59e0b;border-radius:6px} img{width:100%;border-radius:8px;background:#fff}
</style></head><body>
<h1>FedGuard-IoT &mdash; Security Operations Dashboard</h1>
<div class="sub">Global model trained by federated learning across __CLIENTS__ edge sites
 (FedProx + differential privacy). Raw traffic never left the sites. Demo data is synthetic.</div>
<div class="grid">
 <div class="card"><h3>Live detection</h3>
  <button onclick="detect()">Capture random network flow</button>
  <div id="out" style="margin-top:16px;color:#94a3b8">Click the button to inspect a flow.</div></div>
 <div class="card"><h3>Final results (held-out test traffic)</h3>
  <table><tr><th>Setting</th><th>Accuracy</th><th>Macro F1</th></tr>__ROWS__</table></div>
 <div class="card" style="grid-column:1/3"><h3>Training curves &amp; client data distribution</h3>
  <img src="/results.png"></div>
</div>
<script>
async function detect(){
  const r = await (await fetch('/api/detect')).json();
  const cls = r.prediction === 'Normal' ? 'ok' : 'alert';
  const head = r.prediction === 'Normal' ? 'Normal traffic' : 'ALERT: ' + r.prediction;
  const max = Math.max(...r.top_features.map(f => f.impact), 1e-6);
  const rows = r.top_features.map(f => `<tr><td>${f.feature}</td><td style="width:50%">
     <div class="bar" style="width:${Math.max(0, f.impact / max * 100)}%"></div></td>
     <td>${f.impact.toFixed(3)}</td></tr>`).join('');
  document.getElementById('out').innerHTML =
   `<div class="${cls}">${head}</div>
    <div>Confidence ${(r.confidence * 100).toFixed(1)}% &middot; actual label: <b>${r.true_label}</b>
    ${r.prediction === r.true_label ? '&#10004;' : '&#10008;'}</div>
    <h4>Why? Top features driving this decision</h4><table>${rows}</table>`;
}
</script></body></html>"""


@app.route("/")
def index():
    rows = "".join(
        f"<tr><td>{name}</td><td>{h[-1]['accuracy']:.3f}</td><td>{h[-1]['macro_f1']:.3f}</td></tr>"
        for name, h in RESULTS["histories"].items())
    return PAGE.replace("__ROWS__", rows).replace("__CLIENTS__", str(RESULTS["config"]["clients"]))


@app.route("/results.png")
def chart():
    return send_file(os.path.join(HERE, "results.png"))


@app.route("/api/detect")
def detect():
    i = int(rng.integers(len(YTE)))
    e = F.explain(MODEL, XTE[i], BASELINE)
    e["true_label"] = F.CLASSES[int(YTE[i])]
    return jsonify(e)


if __name__ == "__main__":
    app.run(debug=False)
