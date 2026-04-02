import os
import io
import base64
import threading
import time

from flask import Flask, jsonify, render_template_string, send_file

app = Flask(__name__)

_prediction_lock = threading.Lock()
_prediction_running = False
_last_result = None
_last_error = None

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NY Lotto 6/59 Predictor</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #0d1117; color: #e6edf3; min-height: 100vh; }
  header { background: linear-gradient(135deg, #1a3a6b 0%, #0d2137 100%); padding: 28px 24px; text-align: center; border-bottom: 2px solid #21d4fd; }
  header h1 { font-size: 2rem; color: #21d4fd; letter-spacing: 1px; }
  header p { color: #8b949e; margin-top: 6px; font-size: 0.95rem; }
  .container { max-width: 900px; margin: 0 auto; padding: 32px 16px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 24px; margin-bottom: 24px; }
  .card h2 { color: #58a6ff; margin-bottom: 16px; font-size: 1.1rem; border-bottom: 1px solid #21262d; padding-bottom: 10px; }
  .btn { display: inline-block; background: linear-gradient(135deg, #1a6bea 0%, #0d47a1 100%); color: #fff;
         border: none; padding: 14px 36px; border-radius: 8px; font-size: 1rem; font-weight: 600;
         cursor: pointer; transition: opacity 0.2s; }
  .btn:hover { opacity: 0.85; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .ticket { display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; margin: 16px 0; }
  .ball { width: 54px; height: 54px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
          font-size: 1.2rem; font-weight: 700; color: #fff; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }
  .ball-main { background: linear-gradient(135deg, #1565c0 0%, #0d47a1 100%); border: 2px solid #42a5f5; }
  .ball-alt { background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%); border: 2px solid #66bb6a; }
  .info-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }
  .info-item { background: #0d1117; border-radius: 8px; padding: 12px 16px; }
  .info-item .label { font-size: 0.78rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }
  .info-item .value { font-size: 1rem; color: #e6edf3; margin-top: 4px; font-weight: 600; }
  .model-row { display: flex; align-items: flex-start; gap: 10px; padding: 8px 0;
               border-bottom: 1px solid #21262d; }
  .model-row:last-child { border-bottom: none; }
  .model-name { min-width: 140px; color: #8b949e; font-size: 0.88rem; }
  .model-nums { display: flex; flex-wrap: wrap; gap: 6px; }
  .num-badge { background: #21262d; border: 1px solid #30363d; color: #c9d1d9; padding: 2px 8px;
               border-radius: 12px; font-size: 0.82rem; font-family: monospace; }
  .status { padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; font-size: 0.95rem; }
  .status.running { background: #0d2137; border: 1px solid #21d4fd; color: #21d4fd; }
  .status.error { background: #2d1b1b; border: 1px solid #f85149; color: #f85149; }
  .status.success { background: #0d2137; border: 1px solid #3fb950; color: #3fb950; }
  .heatmap img { width: 100%; border-radius: 8px; border: 1px solid #30363d; }
  .lag-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  .lag-table th { color: #8b949e; text-align: left; padding: 6px 10px; font-weight: 500; border-bottom: 1px solid #21262d; }
  .lag-table td { padding: 5px 10px; color: #c9d1d9; }
  .lag-table tr:nth-child(even) td { background: #0d1117; }
  .disclaimer { font-size: 0.78rem; color: #6e7681; background: #161b22; border: 1px solid #21262d;
                border-radius: 8px; padding: 12px 16px; margin-top: 16px; line-height: 1.5; }
  #spinner { display: none; margin-left: 12px; vertical-align: middle; }
</style>
</head>
<body>
<header>
  <h1>NY Lotto 6/59 Ensemble Predictor</h1>
  <p>Markov Chain &bull; Hidden Markov Model &bull; Sum Transitions &bull; Random Forest &bull; Neural Network</p>
</header>
<div class="container">

  <div class="card">
    <h2>Generate Prediction</h2>
    <p style="color:#8b949e; margin-bottom:18px; font-size:0.92rem;">
      Fetches the latest NY Lotto draw history and runs all ensemble models to suggest a ticket.
      This takes ~60&ndash;120 seconds on first run.
    </p>
    <button class="btn" id="runBtn" onclick="runPrediction()">
      Run Prediction
      <svg id="spinner" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4
                 M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83">
          <animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="0.8s" repeatCount="indefinite"/>
        </path>
      </svg>
    </button>
    <div id="statusBox"></div>
  </div>

  <div id="results" style="display:none">
    <div class="card" id="ticketCard">
      <h2>Recommended Ticket</h2>
      <div class="ticket" id="ticketBalls"></div>
      <div class="info-grid" style="margin-top:16px" id="metaInfo"></div>
    </div>

    <div class="card" id="modelsCard">
      <h2>Individual Model Outputs</h2>
      <div id="modelRows"></div>
    </div>

    <div class="card" id="lagsCard">
      <h2>Key Sum Autocorrelation Lags</h2>
      <table class="lag-table">
        <thead><tr><th>Lag</th><th>|Corr|</th><th>Corr</th></tr></thead>
        <tbody id="lagsBody"></tbody>
      </table>
    </div>

    <div class="card heatmap" id="heatmapCard" style="display:none">
      <h2>Markov Transition Heatmap</h2>
      <img id="heatmapImg" src="" alt="Markov transition heatmap">
    </div>
  </div>

  <div class="disclaimer">
    <strong>Disclaimer:</strong> This tool is for entertainment and educational purposes only.
    Lottery outcomes are random and cannot be predicted. Past patterns do not guarantee future results.
    Please gamble responsibly.
  </div>
</div>

<script>
let polling = false;

function runPrediction() {
  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  document.getElementById('spinner').style.display = 'inline';
  setStatus('running', 'Running all models... this may take 60-120 seconds.');
  fetch('/api/predict', { method: 'POST' })
    .then(r => r.json())
    .then(d => {
      if (d.status === 'started' || d.status === 'already_running') {
        pollResults();
      } else {
        setStatus('error', d.error || 'Unknown error starting prediction.');
        resetBtn();
      }
    })
    .catch(e => { setStatus('error', 'Network error: ' + e); resetBtn(); });
}

function pollResults() {
  if (polling) return;
  polling = true;
  const iv = setInterval(() => {
    fetch('/api/results')
      .then(r => r.json())
      .then(d => {
        if (d.status === 'running') {
          // still waiting
        } else if (d.status === 'done') {
          clearInterval(iv);
          polling = false;
          setStatus('success', 'Prediction complete!');
          showResults(d.result);
          resetBtn();
        } else if (d.status === 'error') {
          clearInterval(iv);
          polling = false;
          setStatus('error', 'Error: ' + d.error);
          resetBtn();
        }
      })
      .catch(() => {});
  }, 3000);
}

function resetBtn() {
  document.getElementById('runBtn').disabled = false;
  document.getElementById('spinner').style.display = 'none';
}

function setStatus(type, msg) {
  document.getElementById('statusBox').innerHTML =
    '<div class="status ' + type + '">' + msg + '</div>';
}

function showResults(r) {
  document.getElementById('results').style.display = 'block';

  // Ticket balls
  const tb = document.getElementById('ticketBalls');
  tb.innerHTML = r.final_ticket.map(n =>
    '<div class="ball ball-main">' + n + '</div>').join('');

  // Meta info
  document.getElementById('metaInfo').innerHTML = `
    <div class="info-item"><div class="label">Last Draw Date</div><div class="value">${r.last_date}</div></div>
    <div class="info-item"><div class="label">Last Draw</div><div class="value">${r.last_draw.join(' &bull; ')}</div></div>
    <div class="info-item"><div class="label">Bonus</div><div class="value">${r.last_bonus ?? 'N/A'}</div></div>
    <div class="info-item"><div class="label">Total Draws Used</div><div class="value">${r.total_draws}</div></div>
    <div class="info-item"><div class="label">Ensemble Score</div><div class="value">${r.final_score}</div></div>
    <div class="info-item"><div class="label">Neural Network</div><div class="value">${r.has_torch ? 'Active' : 'Not available'}</div></div>
  `;

  // Model rows
  const mr = document.getElementById('modelRows');
  const models = [
    { name: 'Random Forest', nums: r.rf_pred },
    { name: 'Neural Net', nums: r.nn_pred },
    { name: 'Markov Top-12', nums: r.markov_top12 },
    { name: 'HMM Top-12', nums: r.hmm_top12 },
    { name: 'Sum-Transition Top-12', nums: r.sum_top12 },
  ];
  mr.innerHTML = models.filter(m => m.nums && m.nums.length).map(m =>
    '<div class="model-row"><span class="model-name">' + m.name + '</span>' +
    '<div class="model-nums">' + m.nums.map(n => '<span class="num-badge">' + n + '</span>').join('') +
    '</div></div>'
  ).join('');

  // Lags
  const lb = document.getElementById('lagsBody');
  lb.innerHTML = r.key_lags.map(([lag, absc, corr]) =>
    '<tr><td>' + lag + '</td><td>' + absc + '</td><td>' + corr + '</td></tr>'
  ).join('');

  // Heatmap
  if (r.has_heatmap) {
    const hc = document.getElementById('heatmapCard');
    hc.style.display = 'block';
    document.getElementById('heatmapImg').src = '/api/heatmap?t=' + Date.now();
  }
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/predict", methods=["POST"])
def predict():
    global _prediction_running, _last_result, _last_error

    with _prediction_lock:
        if _prediction_running:
            return jsonify({"status": "already_running"})
        _prediction_running = True
        _last_result = None
        _last_error = None

    def _run():
        global _prediction_running, _last_result, _last_error
        try:
            from main import run_prediction
            result = run_prediction()
            result["has_heatmap"] = bool(result.get("heatmap_path") and
                                         os.path.exists(result["heatmap_path"]))
            _last_result = result
        except Exception as e:
            _last_error = str(e)
        finally:
            _prediction_running = False

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/api/results")
def results():
    global _prediction_running, _last_result, _last_error
    if _prediction_running:
        return jsonify({"status": "running"})
    if _last_error:
        return jsonify({"status": "error", "error": _last_error})
    if _last_result:
        return jsonify({"status": "done", "result": _last_result})
    return jsonify({"status": "idle"})


@app.route("/api/heatmap")
def heatmap():
    from main import HEATMAP_FILE
    if os.path.exists(HEATMAP_FILE):
        return send_file(HEATMAP_FILE, mimetype="image/png")
    return "No heatmap available", 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
