"""
House Price Predictor - Flask app ready for deployment on Render.

Files needed in the repo root:
    app.py
    requirements.txt
    linearregression.pkl

Render settings:
    Build Command : pip install -r requirements.txt
    Start Command : gunicorn app:app
    Env variable  : PYTHON_VERSION = 3.11.9
"""
import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "linearregression.pkl")

app = Flask(__name__)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Order must match the order the model was trained with.
FEATURES = list(model.feature_names_in_)

# label, min, max, step, default (used to build the UI)
FIELDS = {
    "Square_Footage":       ("Living area (sq ft)", 500, 5000, 10, 2000),
    "Num_Bedrooms":         ("Bedrooms", 1, 8, 1, 3),
    "Num_Bathrooms":        ("Bathrooms", 1, 6, 1, 2),
    "Year_Built":           ("Year built", 1900, 2025, 1, 2000),
    "Lot_Size":             ("Lot size (acres)", 0.1, 5, 0.1, 1.0),
    "Garage_Size":          ("Garage (cars)", 0, 4, 1, 1),
    "Neighborhood_Quality": ("Neighborhood quality (1-10)", 1, 10, 1, 5),
}


def predict_price(values: dict):
    row = pd.DataFrame([[float(values[f]) for f in FEATURES]], columns=FEATURES)
    price = float(model.predict(row)[0])
    contributions = {
        f: float(c * float(values[f])) for f, c in zip(FEATURES, model.coef_)
    }
    return price, contributions


@app.route("/")
def home():
    fields = [
        {"key": k, "label": v[0], "min": v[1], "max": v[2], "step": v[3], "default": v[4]}
        for k, v in FIELDS.items()
        if k in FEATURES
    ]
    return render_template_string(PAGE, fields=fields)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or request.form.to_dict()
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify(error=f"Missing fields: {', '.join(missing)}"), 400
    try:
        price, contributions = predict_price(data)
    except (ValueError, TypeError):
        return jsonify(error="All fields must be numbers."), 400
    return jsonify(
        price=round(max(price, 0), 2),
        raw_price=round(price, 2),
        contributions=contributions,
        intercept=float(model.intercept_),
    )


@app.route("/health")
def health():
    return jsonify(status="ok", features=FEATURES)


PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HomeWorth AI - House Price Predictor</title>
<style>
  :root{
    --bg:#f4f6fb; --card:#ffffff; --text:#14213d; --muted:#6b7a99;
    --accent:#4f46e5; --accent2:#06b6d4; --border:#e3e8f2; --bar:#e8ecf7;
    --shadow:0 10px 40px rgba(31,41,100,.10);
  }
  @media (prefers-color-scheme: dark){
    :root{
      --bg:#0b1020; --card:#131a30; --text:#e8ecff; --muted:#93a0c4;
      --border:#222c4a; --bar:#1d2744; --shadow:0 10px 40px rgba(0,0,0,.45);
    }
  }
  *{box-sizing:border-box}
  body{
    margin:0; font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    background:var(--bg); color:var(--text); min-height:100vh;
  }
  .hero{
    background:linear-gradient(135deg,var(--accent),var(--accent2));
    color:#fff; padding:44px 20px 90px; text-align:center;
  }
  .hero h1{margin:0 0 8px; font-size:clamp(1.8rem,4vw,2.6rem); letter-spacing:-.5px}
  .hero p{margin:0; opacity:.9; font-size:1.05rem}
  .wrap{
    max-width:1050px; margin:-60px auto 40px; padding:0 18px;
    display:grid; grid-template-columns:1.3fr 1fr; gap:22px;
  }
  @media(max-width:820px){.wrap{grid-template-columns:1fr}}
  .card{
    background:var(--card); border:1px solid var(--border); border-radius:20px;
    padding:26px; box-shadow:var(--shadow);
  }
  .card h2{margin:0 0 18px; font-size:1.1rem}
  .field{margin-bottom:18px}
  .row{display:flex; justify-content:space-between; align-items:center; margin-bottom:6px}
  .row label{font-size:.92rem; color:var(--muted); font-weight:600}
  .row input[type=number]{
    width:96px; padding:6px 8px; border-radius:8px; border:1px solid var(--border);
    background:var(--bg); color:var(--text); font-weight:700; text-align:right;
  }
  input[type=range]{width:100%; accent-color:var(--accent)}
  .result{text-align:center; position:sticky; top:20px}
  .tag{color:var(--muted); font-size:.85rem; text-transform:uppercase; letter-spacing:1.5px}
  .price{
    font-size:clamp(2.2rem,5vw,3.2rem); font-weight:800; margin:8px 0 4px;
    background:linear-gradient(135deg,var(--accent),var(--accent2));
    -webkit-background-clip:text; background-clip:text; color:transparent;
  }
  .sub{color:var(--muted); font-size:.9rem; margin-bottom:22px}
  .drivers{text-align:left; margin-top:8px}
  .drivers h3{font-size:.95rem; margin:0 0 12px}
  .d{margin-bottom:11px}
  .d .n{display:flex; justify-content:space-between; font-size:.82rem; color:var(--muted); margin-bottom:4px}
  .track{height:8px; background:var(--bar); border-radius:99px; overflow:hidden}
  .fill{height:100%; border-radius:99px; background:linear-gradient(90deg,var(--accent),var(--accent2)); transition:width .4s}
  .btn{
    margin-top:18px; width:100%; padding:13px; border:0; border-radius:12px; cursor:pointer;
    font-weight:700; font-size:1rem; color:#fff;
    background:linear-gradient(135deg,var(--accent),var(--accent2));
  }
  .btn:hover{filter:brightness(1.08)}
  .err{color:#e11d48; font-size:.9rem; margin-top:10px; min-height:1em}
  footer{text-align:center; color:var(--muted); font-size:.8rem; padding:0 0 30px}
</style>
</head>
<body>
<div class="hero">
  <h1>🏡 HomeWorth AI</h1>
  <p>Instant house price estimates powered by a Linear Regression model</p>
</div>

<div class="wrap">
  <div class="card">
    <h2>Property details</h2>
    <div id="fields">
    {% for f in fields %}
      <div class="field">
        <div class="row">
          <label for="n_{{ f.key }}">{{ f.label }}</label>
          <input type="number" id="n_{{ f.key }}" data-key="{{ f.key }}"
                 min="{{ f.min }}" max="{{ f.max }}" step="{{ f.step }}" value="{{ f.default }}">
        </div>
        <input type="range" id="r_{{ f.key }}" data-key="{{ f.key }}"
               min="{{ f.min }}" max="{{ f.max }}" step="{{ f.step }}" value="{{ f.default }}">
      </div>
    {% endfor %}
    </div>
  </div>

  <div class="card result">
    <div class="tag">Estimated market value</div>
    <div class="price" id="price">-</div>
    <div class="sub">Updates live as you change the details</div>
    <div class="drivers">
      <h3>What drives this price</h3>
      <div id="drivers"></div>
    </div>
    <button class="btn" id="go">Predict price</button>
    <div class="err" id="err"></div>
  </div>
</div>
<footer>Estimates are for guidance only and not a formal property valuation.</footer>

<script>
const labels = {{ fields | map(attribute='key') | list | tojson }};
const nice = {{ (fields | map(attribute='label') | list) | tojson }};
const names = Object.fromEntries(labels.map((k,i)=>[k,nice[i]]));
const money = new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:0});
let timer;

function values(){
  const v = {};
  document.querySelectorAll('input[type=number]').forEach(i => v[i.dataset.key] = parseFloat(i.value));
  return v;
}

function animate(el, to){
  const from = parseFloat(el.dataset.v || 0), t0 = performance.now(), dur = 450;
  function step(t){
    const p = Math.min((t - t0) / dur, 1), e = 1 - Math.pow(1 - p, 3);
    el.textContent = money.format(from + (to - from) * e);
    if(p < 1) requestAnimationFrame(step); else el.dataset.v = to;
  }
  requestAnimationFrame(step);
}

async function predict(){
  const err = document.getElementById('err'); err.textContent = '';
  try{
    const res = await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(values())});
    const d = await res.json();
    if(!res.ok) throw new Error(d.error || 'Request failed');
    animate(document.getElementById('price'), d.price);
    const entries = Object.entries(d.contributions).sort((a,b)=>b[1]-a[1]);
    const max = Math.max(...entries.map(e=>Math.abs(e[1])), 1);
    document.getElementById('drivers').innerHTML = entries.map(([k,c]) =>
      `<div class="d"><div class="n"><span>${names[k]||k}</span><span>${money.format(c)}</span></div>
       <div class="track"><div class="fill" style="width:${Math.max(Math.abs(c)/max*100,2)}%"></div></div></div>`).join('');
  }catch(e){ err.textContent = e.message; }
}

function queue(){ clearTimeout(timer); timer = setTimeout(predict, 200); }

document.querySelectorAll('input[type=range]').forEach(r => r.addEventListener('input', () => {
  document.getElementById('n_' + r.dataset.key).value = r.value; queue();
}));
document.querySelectorAll('input[type=number]').forEach(n => n.addEventListener('input', () => {
  document.getElementById('r_' + n.dataset.key).value = n.value; queue();
}));
document.getElementById('go').addEventListener('click', predict);
predict();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
