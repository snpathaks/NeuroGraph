"""
server.py — NeuroGraph Flask Backend
======================================
Serves the HTML/CSS/JS dashboard and exposes REST API endpoints
that perform real ML inference using the trained RF and XGBoost models.

Run:
    python server.py
Then open: http://localhost:5000
"""

import os
import sys
import json
import pickle
import warnings
import tempfile

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from feature_extraction import (
    extract_features,
    _velocity_acceleration_features,
    _tremor_features,
    _pressure_velocity_coupling,
    _adaptive_segment_features,
    _directional_entropy,
    _pen_lift_ratio,
)
from utils import normalize_stroke, resample_stroke

# ── Constants ─────────────────────────────────────────────────────────────────
RF_PATH    = os.path.join(ROOT, "models", "neurograph_rf.pkl")
XGB_PATH   = os.path.join(ROOT, "models", "neurograph_xgb.pkl")
CACHE_PATH = os.path.join(ROOT, "models", "features_cache.csv")

FEATURE_COLS = [
    "speed_mean","speed_std","speed_max","speed_iqr",
    "accel_mean","accel_std","accel_max","accel_iqr",
    "jerk_mean","jerk_std","jerk_max","jerk_iqr",
    "tremor_ratio_vx","tremor_peak_hz_vx",
    "tremor_ratio_vy","tremor_peak_hz_vy",
    "tremor_ratio_combined","pressure_speed_corr",
    "seg0_jerk_var","seg0_weight","seg1_jerk_var","seg1_weight",
    "seg2_jerk_var","seg2_weight","seg3_jerk_var","seg3_weight",
    "seg4_jerk_var","seg4_weight","weighted_jerk_var_mean",
    "directional_entropy","pen_lift_ratio",
]

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=ROOT, static_url_path="")


# ── Model loading ─────────────────────────────────────────────────────────────
_models = {}

def get_models():
    global _models
    if not _models:
        for name, path, key in [
            ("Random Forest", RF_PATH, "rf"),
            ("XGBoost",       XGB_PATH, "xgb"),
        ]:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    _models[key] = {"label": name, "model": pickle.load(f)}
    return _models


def get_cache():
    if os.path.exists(CACHE_PATH):
        return pd.read_csv(CACHE_PATH)
    return pd.DataFrame()


# ── Prediction helper ─────────────────────────────────────────────────────────
def predict(feature_dict: dict, model) -> dict:
    row  = [feature_dict.get(c, 0.0) for c in FEATURE_COLS]
    X    = np.array(row, dtype=float).reshape(1, -1)
    X    = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    prob = model.predict_proba(X)[0]
    pred = model.predict(X)[0]
    pd_prob   = float(prob[1])
    ctrl_prob = float(prob[0])
    label     = "Parkinson's" if pred == 1 else "Control"
    return {
        "label":     label,
        "is_pd":     bool(pred == 1),
        "pd_prob":   round(pd_prob * 100, 1),
        "ctrl_prob": round(ctrl_prob * 100, 1),
        "jerk_mean":              round(float(feature_dict.get("jerk_mean", 0)), 4),
        "tremor_ratio_combined":  round(float(feature_dict.get("tremor_ratio_combined", 0)), 5),
        "tremor_peak_hz_vx":      round(float(feature_dict.get("tremor_peak_hz_vx", 0)), 2),
        "directional_entropy":    round(float(feature_dict.get("directional_entropy", 0)), 3),
        "pen_lift_ratio":         round(float(feature_dict.get("pen_lift_ratio", 0)), 3),
        "speed_mean":             round(float(feature_dict.get("speed_mean", 0)), 3),
    }


# ── Canvas points → stroke DataFrame ─────────────────────────────────────────
def points_to_stroke_df(points: list) -> pd.DataFrame | None:
    """Convert [{x, y, t}, ...] from the JS canvas into a stroke DataFrame."""
    if not points or len(points) < 10:
        return None
    rows = []
    t0 = points[0].get("t", 0)
    for pt in points:
        rows.append({
            "x":             float(pt.get("x", 0)),
            "y":             float(pt.get("y", 0)),
            "timestamp":     float(pt.get("t", 0)) - t0,
            "pressure":      float(pt.get("pressure", 512)),
            "button_status": int(pt.get("button", 1)),
        })
    return pd.DataFrame(rows)


def features_from_stroke_df(stroke_df: pd.DataFrame) -> dict | None:
    """Run normalise → resample → extract pipeline on an in-memory DataFrame."""
    try:
        norm  = normalize_stroke(stroke_df)
        proc  = resample_stroke(norm, n_points=500)
        va    = _velocity_acceleration_features(proc)
        vx    = va.pop("_vx");  vy = va.pop("_vy")
        speed = va.pop("_speed"); dt = va.pop("_dt")
        feats = dict(va)
        feats.update(_tremor_features(vx, vy, dt))
        feats.update(_pressure_velocity_coupling(proc, speed))
        feats.update(_adaptive_segment_features(proc))
        feats.update(_directional_entropy(proc))
        feats.update(_pen_lift_ratio(stroke_df))
        return feats
    except Exception as e:
        print(f"[NeuroGraph] Feature extraction error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Routes — Static
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(ROOT, "dashboard.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(ROOT, filename)


# ═══════════════════════════════════════════════════════════════════════════════
# Routes — API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/models")
def api_models():
    """List available models."""
    models = get_models()
    return jsonify({"models": [{"key": k, "label": v["label"]} for k, v in models.items()]})


@app.route("/api/analyse/draw", methods=["POST"])
def api_analyse_draw():
    """
    Analyse canvas stroke points.
    Body: { "points": [{x, y, t, pressure?, button?}, ...], "model": "rf"|"xgb" }
    """
    data       = request.get_json(force=True)
    raw_points = data.get("points", [])
    model_key  = data.get("model", "rf")

    models = get_models()
    if not models:
        return jsonify({"error": "No trained models found. Run model.py first."}), 500
    if model_key not in models:
        model_key = next(iter(models))

    if len(raw_points) < 20:
        return jsonify({"error": "Too few points — draw a longer stroke."}), 400

    stroke_df = points_to_stroke_df(raw_points)
    if stroke_df is None:
        return jsonify({"error": "Could not parse stroke data."}), 400

    feats = features_from_stroke_df(stroke_df)
    if feats is None:
        return jsonify({"error": "Feature extraction failed."}), 500

    result = predict(feats, models[model_key]["model"])
    result["model_used"] = models[model_key]["label"]
    return jsonify(result)


@app.route("/api/analyse/upload", methods=["POST"])
def api_analyse_upload():
    """
    Analyse an uploaded stroke .txt file.
    Form data: file=<file>, model=rf|xgb, test_id=<int>|-1
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    f         = request.files["file"]
    model_key = request.form.get("model", "rf")
    test_id   = request.form.get("test_id", "-1")
    try:
        test_id_filter = int(test_id) if test_id != "-1" else None
    except ValueError:
        test_id_filter = None

    models = get_models()
    if not models:
        return jsonify({"error": "No trained models found."}), 500
    if model_key not in models:
        model_key = next(iter(models))

    # Save to temp file (extract_features expects a path)
    suffix = os.path.splitext(f.filename)[1] or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=os.path.join(ROOT, "models")) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        feats = extract_features(tmp_path, test_id_filter=test_id_filter)
        if feats is None:
            return jsonify({"error": "Feature extraction returned no data."}), 400
    except Exception as e:
        return jsonify({"error": f"Feature extraction failed: {e}"}), 500
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # Count rows for display
    try:
        with open(tmp_path) as _:
            pass
    except Exception:
        pass

    result = predict(feats, models[model_key]["model"])
    result["model_used"]  = models[model_key]["label"]
    result["filename"]    = f.filename
    result["row_count"]   = None   # file already deleted; row count parsed in JS
    return jsonify(result)


@app.route("/api/dataset")
def api_dataset():
    """Return features_cache.csv as JSON for the Dataset Explorer."""
    cache_df = get_cache()
    if cache_df.empty:
        return jsonify({"error": "No cached features. Run model.py first.", "rows": []})

    display_cols = [
        "subject_id", "label",
        "jerk_mean", "tremor_ratio_combined",
        "directional_entropy", "pen_lift_ratio", "pressure_speed_corr",
    ]
    available = [c for c in display_cols if c in cache_df.columns]
    subset    = cache_df[available].copy()

    # Round floats for cleaner JSON
    for col in subset.select_dtypes(include="float").columns:
        subset[col] = subset[col].round(5)

    records = subset.to_dict(orient="records")
    stats = {
        "total":    int(len(cache_df)),
        "pd":       int(cache_df["label"].sum()) if "label" in cache_df.columns else 0,
        "control":  int((cache_df["label"] == 0).sum()) if "label" in cache_df.columns else 0,
        "features": len(FEATURE_COLS),
    }
    return jsonify({"rows": records, "stats": stats})


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    models = get_models()
    if not models:
        print("[NeuroGraph] WARNING: No trained models found in models/. Run model.py first.")
    else:
        print(f"[NeuroGraph] Loaded models: {', '.join(v['label'] for v in models.values())}")

    print("[NeuroGraph] Starting server at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
