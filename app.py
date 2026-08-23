"""
app.py — NeuroGraph Streamlit Application
==========================================
A clinical-style Parkinson's detection interface with two modes:

  Tab 1 — Live Draw   : draw a spiral on the canvas; simulate pen data;
                        run feature extraction; show prediction.
  Tab 2 — Upload CSV  : upload a raw stroke .txt file (same format as the
                        UCI dataset); extract features; show prediction.
  Tab 3 — Dataset Explorer : browse the cached features, compare PD vs Control.

Run:
    streamlit run app.py
"""

import os
import sys
import pickle
import warnings
import time

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from streamlit_drawable_canvas import st_canvas

# ---- Path setup so imports work when launched from any CWD ----------------
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from feature_extraction import extract_features, _velocity_acceleration_features
from utils import normalize_stroke, resample_stroke

# ---- Constants --------------------------------------------------------------
RF_PATH  = os.path.join(ROOT, "models", "neurograph_rf.pkl")
XGB_PATH = os.path.join(ROOT, "models", "neurograph_xgb.pkl")
CACHE_PATH = os.path.join(ROOT, "models", "features_cache.csv")

META_COLS    = {"subject_id", "label", "filepath"}
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

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroGraph — Parkinson's Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Base text: pure white everywhere ── */
.stApp { background: #0d1117; color: #ffffff !important; }

/* ── Streamlit top header bar: dark bg, black text ── */
header[data-testid="stHeader"],
header[data-testid="stHeader"] * {
    background-color: #000000 !important;
    color: #000000 !important;
}
/* Deploy button and toolbar icons in the header */
header[data-testid="stHeader"] button,
header[data-testid="stHeader"] a,
header[data-testid="stHeader"] svg,
header[data-testid="stHeader"] span {
    color: #000000 !important;
    fill: #000000 !important;
}

p, span, div, label, li, td, th, caption,
.stMarkdown, .stText, .stCaption,
[class*="stMarkdown"] p,
[class*="stMarkdown"] li,
[class*="stMarkdown"] span { color: #ffffff !important; }

/* Headings */
h1, h2, h3, h4, h5, h6,
[class*="stMarkdown"] h1,
[class*="stMarkdown"] h2,
[class*="stMarkdown"] h3 { color: #ffffff !important; font-weight: 700; }

/* Streamlit native caption / small text */
.stCaption, [data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p { color: #cccccc !important; }

/* Sidebar */
[data-testid="stSidebar"],
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    color: #ffffff !important;
}
[data-testid="stSidebar"] { border-right: 1px solid #30363d; }

/* Selectbox / radio / widget labels */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
.stSelectbox label, .stRadio label,
.stFileUploader label { color: #ffffff !important; font-weight: 500; }

/* ── Selectbox: closed input box ── */
[data-baseweb="select"] {
    background-color: #21262d !important;
}
[data-baseweb="select"] > div,
[data-baseweb="select"] input,
[data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
[data-baseweb="select"] span,
[data-baseweb="select"] div[aria-selected] {
    background-color: #21262d !important;
    color: #ffffff !important;
}
/* Selected value text inside the closed box */
[data-baseweb="select"] [data-testid="stMarkdownContainer"],
[data-baseweb="select"] .css-1dimb5e-singleValue,
[data-baseweb="select"] [class*="singleValue"],
[data-baseweb="select"] [class*="placeholder"],
[data-baseweb="select"] [class*="ValueContainer"] *,
[data-baseweb="select"] [class*="Input"] *,
[data-baseweb="select"] p {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
/* ── Selectbox: open dropdown list ── */
[data-baseweb="popover"],
[data-baseweb="popover"] ul,
[data-baseweb="menu"],
[data-baseweb="menu"] ul,
[role="listbox"],
[role="option"] {
    background-color: #21262d !important;
    color: #ffffff !important;
}
[role="option"] span,
[data-baseweb="option"],
[data-baseweb="option"] span,
[data-baseweb="option"] div {
    background-color: #21262d !important;
    color: #ffffff !important;
}
[data-baseweb="option"]:hover,
[role="option"]:hover {
    background-color: #30363d !important;
    color: #ffffff !important;
}

/* Radio option text */
[data-testid="stRadio"] label span { color: #ffffff !important; }

/* Tab labels */
[data-testid="stTab"] button,
[data-testid="stTab"] button p,
button[data-baseweb="tab"] { color: #ffffff !important; font-weight: 500; }

/* Info / warning / success boxes */
[data-testid="stAlert"] p,
.stAlert p { color: #ffffff !important; }

/* Expander header */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p { color: #ffffff !important; font-weight: 600; }

/* Dataframe / table text */
[data-testid="stDataFrame"] td,
[data-testid="stDataFrame"] th,
.dataframe td, .dataframe th { color: #ffffff !important; }

/* Metric (st.metric) */
[data-testid="stMetric"] label,
[data-testid="stMetricLabel"] p,
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] p { color: #ffffff !important; }

/* Cards */
.ng-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    color: #ffffff;
}

/* Result banners */
.result-pd {
    background: linear-gradient(135deg, #3d0000 0%, #6b0000 100%);
    border: 2px solid #ff4444;
    border-radius: 14px;
    padding: 22px 28px;
    text-align: center;
}
.result-ctrl {
    background: linear-gradient(135deg, #003d1a 0%, #006b2e 100%);
    border: 2px solid #00cc66;
    border-radius: 14px;
    padding: 22px 28px;
    text-align: center;
}
.result-title { font-size: 2rem; font-weight: 700; margin: 0; color: #ffffff !important; }
.result-sub   { font-size: 1rem; color: #ffffff !important; opacity: 0.92; margin-top: 6px; }

/* Metric tiles */
.metric-grid { display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0; }
.metric-tile {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px 18px;
    min-width: 130px;
    flex: 1;
}
.metric-tile .val { font-size: 1.5rem; font-weight: 600; color: #58a6ff !important; }
.metric-tile .lbl { font-size: 0.75rem; color: #cccccc !important; margin-top: 2px; }

/* Section headers */
.section-header {
    font-size: 1.1rem; font-weight: 600; color: #ffffff !important;
    border-left: 3px solid #58a6ff;
    padding-left: 10px; margin: 20px 0 12px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #238636, #2ea043);
    color: #ffffff !important; border: none; border-radius: 8px;
    font-weight: 600; padding: 8px 20px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2ea043, #3fb950);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(46,160,67,0.4);
}

/* Divider */
hr { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)


# ── Model loading (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    for name, path in [("Random Forest", RF_PATH), ("XGBoost", XGB_PATH)]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    return models


@st.cache_data(show_spinner=False)
def load_cache():
    if os.path.exists(CACHE_PATH):
        return pd.read_csv(CACHE_PATH)
    return pd.DataFrame()


# ── Prediction helper ────────────────────────────────────────────────────────
def predict(feature_dict: dict, model) -> tuple[str, float, float]:
    """Returns (label_str, pd_probability, ctrl_probability)."""
    row = [feature_dict.get(c, 0.0) for c in FEATURE_COLS]
    X   = np.array(row, dtype=float).reshape(1, -1)
    X   = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    prob = model.predict_proba(X)[0]
    pred = model.predict(X)[0]
    label = "Parkinson's" if pred == 1 else "Control (Healthy)"
    pd_prob   = float(prob[1])
    ctrl_prob = float(prob[0])
    return label, pd_prob, ctrl_prob


# ── Feature radar chart ──────────────────────────────────────────────────────
def radar_chart(feature_dict: dict, cache_df: pd.DataFrame) -> go.Figure:
    """Overlay subject features against dataset mean (normalised 0-1)."""
    key_feats = [
        "jerk_mean", "speed_mean", "tremor_ratio_combined",
        "directional_entropy", "pen_lift_ratio", "pressure_speed_corr",
        "weighted_jerk_var_mean", "accel_mean",
    ]
    labels = [
        "Jerk", "Speed", "Tremor", "Dir. Entropy",
        "Pen Lift", "Pressure Coupling", "Wtd Jerk Var", "Accel",
    ]

    subject_vals = np.array([abs(feature_dict.get(f, 0)) for f in key_feats],
                             dtype=float)

    if not cache_df.empty:
        means = cache_df[key_feats].mean().values.astype(float)
        pd_means = cache_df[cache_df["label"]==1][key_feats].mean().values.astype(float)
        ctrl_means = cache_df[cache_df["label"]==0][key_feats].mean().values.astype(float)
    else:
        means = np.ones(len(key_feats))
        pd_means = ctrl_means = means

    # Normalise all to [0,1] using dataset max
    all_vals = np.vstack([subject_vals, pd_means, ctrl_means])
    col_max  = all_vals.max(axis=0)
    col_max[col_max == 0] = 1
    subject_n  = subject_vals  / col_max
    pd_n       = pd_means      / col_max
    ctrl_n     = ctrl_means    / col_max

    fig = go.Figure()
    for vals, name, color, dash in [
        (ctrl_n,    "Dataset: Control",   "#00cc66", "dash"),
        (pd_n,      "Dataset: PD",        "#ff6b6b", "dash"),
        (subject_n, "This Sample",        "#58a6ff", "solid"),
    ]:
        v = list(vals) + [vals[0]]
        l = labels + [labels[0]]
        fig.add_trace(go.Scatterpolar(
            r=v, theta=l, name=name, fill="toself" if name=="This Sample" else None,
            line=dict(color=color, width=2, dash=dash),
            opacity=0.85,
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="#161b22",
            radialaxis=dict(visible=True, range=[0,1],
                            gridcolor="#30363d", color="#8b949e"),
            angularaxis=dict(gridcolor="#30363d", color="#e6edf3"),
        ),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3", family="Inter"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        margin=dict(t=30, b=30),
        height=360,
    )
    return fig


# ── Stroke time-series chart ─────────────────────────────────────────────────
def stroke_chart(df: pd.DataFrame) -> go.Figure:
    """Velocity and pressure over time for a processed stroke."""
    t = df["timestamp"].values * 1e-3
    x = df["x"].values
    y = df["y"].values
    dt = np.diff(t)
    dt[dt == 0] = 1e-9
    vx = np.diff(x) / dt
    vy = np.diff(y) / dt
    speed = np.sqrt(vx**2 + vy**2)
    pressure = df["pressure"].values[:-1]
    t_mid = (t[:-1] + t[1:]) / 2

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_mid, y=speed, name="Speed",
                             line=dict(color="#58a6ff", width=1.5)))
    fig.add_trace(go.Scatter(x=t_mid, y=pressure, name="Pressure",
                             line=dict(color="#ff9944", width=1.5),
                             yaxis="y2"))
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font=dict(color="#e6edf3", family="Inter"),
        xaxis=dict(title="Time (s)", gridcolor="#21262d"),
        yaxis=dict(title="Speed (coord/s)", gridcolor="#21262d", color="#58a6ff"),
        yaxis2=dict(title="Pressure (norm.)", overlaying="y", side="right",
                    color="#ff9944"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        margin=dict(t=20, b=40, l=60, r=60),
        height=260,
    )
    return fig


# ── Canvas → synthetic stroke DataFrame ─────────────────────────────────────
def canvas_to_stroke_df(canvas_result) -> pd.DataFrame | None:
    """
    Convert the drawable canvas JSON path data into a synthetic stroke
    DataFrame that matches the format expected by feature_extraction.
    """
    if canvas_result is None or canvas_result.json_data is None:
        return None

    objects = canvas_result.json_data.get("objects", [])
    if not objects:
        return None

    rows = []
    t = 0
    for obj in objects:
        if obj.get("type") != "path":
            continue
        path_data = obj.get("path", [])
        for cmd in path_data:
            if not cmd or cmd[0] not in ("M", "L", "Q"):
                continue
            if cmd[0] == "M" and len(cmd) >= 3:
                rows.append({"x": cmd[1], "y": cmd[2], "t": t,
                             "pressure": 0.5, "button_status": 1})
                t += 7
            elif cmd[0] == "L" and len(cmd) >= 3:
                rows.append({"x": cmd[1], "y": cmd[2], "t": t,
                             "pressure": 0.5, "button_status": 1})
                t += 7
            elif cmd[0] == "Q" and len(cmd) >= 5:
                rows.append({"x": cmd[3], "y": cmd[4], "t": t,
                             "pressure": 0.5, "button_status": 1})
                t += 7

    if len(rows) < 10:
        return None

    df = pd.DataFrame(rows)
    df.rename(columns={"t": "timestamp"}, inplace=True)
    return df


def features_from_stroke_df(stroke_df: pd.DataFrame) -> dict | None:
    """
    Run the same normalise → resample → extract pipeline used in
    feature_extraction.py, but directly on an in-memory DataFrame.
    """
    try:
        norm = normalize_stroke(stroke_df)
        proc = resample_stroke(norm, n_points=500)

        # velocity / acceleration / jerk
        va = _velocity_acceleration_features(proc)
        vx    = va.pop("_vx");  vy = va.pop("_vy")
        speed = va.pop("_speed"); dt = va.pop("_dt")
        feats = dict(va)

        # tremor
        from feature_extraction import (
            _tremor_features, _pressure_velocity_coupling,
            _adaptive_segment_features, _directional_entropy, _pen_lift_ratio,
        )
        feats.update(_tremor_features(vx, vy, dt))
        feats.update(_pressure_velocity_coupling(proc, speed))
        feats.update(_adaptive_segment_features(proc))
        feats.update(_directional_entropy(proc))
        feats.update(_pen_lift_ratio(stroke_df))
        return feats
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return None


# ── Result display ────────────────────────────────────────────────────────────
def show_result(label: str, pd_prob: float, ctrl_prob: float,
                feats: dict, cache_df: pd.DataFrame) -> None:

    is_pd = "Parkinson" in label
    css   = "result-pd" if is_pd else "result-ctrl"
    icon  = "⚠️" if is_pd else "✅"
    conf  = pd_prob if is_pd else ctrl_prob

    st.markdown(f"""
    <div class="{css}">
      <p class="result-title">{icon} {label}</p>
      <p class="result-sub">Model confidence: {conf*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    for lbl, val in [
        ("PD Probability",    f"{pd_prob*100:.1f}%"),
        ("Control Prob.",     f"{ctrl_prob*100:.1f}%"),
        ("Jerk Mean",         f"{feats.get('jerk_mean',0):.3f}"),
        ("Tremor Ratio",      f"{feats.get('tremor_ratio_combined',0):.4f}"),
        ("Dir. Entropy",      f"{feats.get('directional_entropy',0):.3f}"),
        ("Pen Lift Ratio",    f"{feats.get('pen_lift_ratio',0):.2f}"),
    ]:
        st.markdown(f"""
        <div class="metric-tile">
          <div class="val">{val}</div>
          <div class="lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Radar
    st.markdown('<div class="section-header">Feature Profile vs Dataset</div>',
                unsafe_allow_html=True)
    st.plotly_chart(radar_chart(feats, cache_df),
                    use_container_width=True, key="radar")

    # Disclaimer
    st.info("⚕️ **Clinical disclaimer**: This tool is a research prototype. "
            "Results must not be used for clinical diagnosis.")


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧠 NeuroGraph")
    st.caption("Parkinson's Detection via Stroke Analysis")
    st.divider()

    models = load_models()
    if not models:
        st.error("No trained models found. Run `python model.py` first.")
        st.stop()

    model_name = st.selectbox("Model", list(models.keys()))
    model      = models[model_name]

    st.divider()
    st.markdown("**About**")
    st.caption(
        "NeuroGraph analyses handwriting stroke dynamics — velocity, jerk, "
        "tremor frequency, and pressure coupling — to screen for early "
        "Parkinson's disease indicators."
    )
    st.divider()
    st.caption("Dataset: UCI Spiral Drawing Tablet · 40 subjects")
    st.caption(f"Model: {model_name}")

cache_df = load_cache()

# ═══════════════════════════════════════════════════════════════════════════════
# Main tabs
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "🖊️ Live Draw",
    "📂 Upload Stroke File",
    "📊 Dataset Explorer",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Live Draw
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Draw a spiral below")
    st.caption(
        "Draw a spiral (or any continuous stroke) — the app will extract "
        "movement features and predict PD likelihood in real time."
    )

    col_canvas, col_result = st.columns([1.1, 0.9], gap="large")

    with col_canvas:
        st.markdown('<div class="ng-card">', unsafe_allow_html=True)

        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=3,
            stroke_color="#58a6ff",
            background_color="#161b22",
            height=400,
            width=460,
            drawing_mode="freedraw",
            key="canvas",
            display_toolbar=True,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            analyse_btn = st.button("🔍 Analyse Stroke", use_container_width=True)
        with col_b:
            if st.button("🗑️ Clear Canvas", use_container_width=True):
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        if analyse_btn:
            stroke_df = canvas_to_stroke_df(canvas_result)

            if stroke_df is None or len(stroke_df) < 10:
                st.warning("Please draw a longer stroke before analysing.")
            else:
                with st.spinner("Extracting features…"):
                    time.sleep(0.3)
                    feats = features_from_stroke_df(stroke_df)

                if feats:
                    label, pd_prob, ctrl_prob = predict(feats, model)
                    show_result(label, pd_prob, ctrl_prob, feats, cache_df)

                    # Velocity / pressure time-series
                    st.markdown('<div class="section-header">Stroke Signal</div>',
                                unsafe_allow_html=True)
                    try:
                        norm = normalize_stroke(stroke_df)
                        proc = resample_stroke(norm, 500)
                        st.plotly_chart(stroke_chart(proc),
                                        use_container_width=True, key="ts_draw")
                    except Exception:
                        pass
        else:
            st.markdown('<div class="ng-card">', unsafe_allow_html=True)
            st.markdown("""
            **Instructions**

            1. Draw a **spiral** on the canvas (start from centre, spiral outward)
            2. Try to keep it smooth — the model will detect tremor/jerk
            3. Click **Analyse Stroke** when done
            4. Results and feature profile appear here

            ---
            **What the model looks at:**
            - 🌀 **Jerk** — sudden direction changes (high in PD)
            - 📈 **Tremor frequency** — 3–12 Hz oscillation in velocity
            - 🖊️ **Directional entropy** — chaotic vs smooth stroke direction
            - ⬇️ **Pressure–velocity coupling** — breaks down in PD
            """)
            st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Upload CSV
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Upload a raw stroke file")
    st.caption(
        "Upload a `.txt` file from the UCI Parkinson spiral dataset "
        "(semicolon-separated, 7 columns: X;Y;ButtonStatus;Pressure;GripAngle;Timestamp;TestID)."
    )

    col_up, col_res2 = st.columns([1, 1], gap="large")

    with col_up:
        uploaded = st.file_uploader("Choose a .txt stroke file",
                                     type=["txt", "csv"])

        test_id_choice = st.radio(
            "Test type filter",
            ["All tests (no filter)", "Static Spiral (0)",
             "Dynamic Spiral (1)", "Stability Test (2)"],
            help="Filter rows by TestID column before feature extraction"
        )
        tid_map = {
            "All tests (no filter)": None,
            "Static Spiral (0)": 0,
            "Dynamic Spiral (1)": 1,
            "Stability Test (2)": 2,
        }
        test_id_filter = tid_map[test_id_choice]

        if uploaded:
            # Save to a temp file (feature_extraction expects a file path)
            tmp_path = os.path.join(ROOT, "models", "_tmp_upload.txt")
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getvalue())

            with st.spinner("Extracting features from uploaded file…"):
                try:
                    feats = extract_features(tmp_path,
                                             test_id_filter=test_id_filter)
                    st.success(f"Extracted {len(feats)-1} features successfully.")
                except Exception as e:
                    feats = None
                    st.error(f"Feature extraction failed: {e}")
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    with col_res2:
        if uploaded and feats:
            label, pd_prob, ctrl_prob = predict(feats, model)
            show_result(label, pd_prob, ctrl_prob, feats, cache_df)

            # Show raw feature table
            st.markdown('<div class="section-header">Raw Feature Values</div>',
                        unsafe_allow_html=True)
            feat_df = pd.DataFrame([
                {"Feature": k, "Value": round(v, 6)}
                for k, v in feats.items()
                if isinstance(v, float) and k not in META_COLS
            ])
            st.dataframe(feat_df, use_container_width=True, height=280)
        elif not uploaded:
            st.markdown('<div class="ng-card">', unsafe_allow_html=True)
            st.markdown("""
            **Expected file format**

            Semicolon-separated, no header row:
            ```
            199;203;0;1;990;14431033;0
            199;203;0;9;970;14431040;0
            ...
            ```

            Columns (in order):
            | Col | Meaning |
            |-----|---------|
            | 0   | X coordinate |
            | 1   | Y coordinate |
            | 2   | Button/Z status |
            | 3   | Pressure |
            | 4   | Grip angle |
            | 5   | Timestamp (ms) |
            | 6   | Test ID (0/1/2) |
            """)
            st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Dataset Explorer
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Dataset Feature Explorer")

    if cache_df.empty:
        st.warning("No cached features found. Run `python model.py` first.")
    else:
        n_pd   = int(cache_df["label"].sum())
        n_ctrl = int((cache_df["label"] == 0).sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Subjects", len(cache_df))
        m2.metric("Parkinson's",    n_pd)
        m3.metric("Control",        n_ctrl)
        m4.metric("Features",       len(FEATURE_COLS))

        st.divider()

        col_feat, col_box = st.columns([1, 2], gap="large")

        with col_feat:
            st.markdown("**Select feature to explore**")
            selected_feat = st.selectbox(
                "Feature", FEATURE_COLS,
                index=FEATURE_COLS.index("jerk_mean"),
                label_visibility="collapsed",
            )

        with col_box:
            plot_df = cache_df[["label", selected_feat]].copy()
            plot_df["Group"] = plot_df["label"].map(
                {1: "Parkinson's", 0: "Control"}
            )
            fig_box = px.box(
                plot_df, x="Group", y=selected_feat,
                color="Group",
                color_discrete_map={"Parkinson's": "#ff6b6b", "Control": "#00cc66"},
                points="all",
                template="plotly_dark",
                title=f"Distribution: {selected_feat}",
            )
            fig_box.update_layout(
                paper_bgcolor="#0d1117",
                plot_bgcolor="#161b22",
                font=dict(color="#e6edf3", family="Inter"),
                showlegend=False,
                height=340,
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_box, use_container_width=True, key="boxplot")

        st.divider()

        # Feature correlation heatmap (PD vs Control mean difference)
        st.markdown('<div class="section-header">Mean Feature Values: PD vs Control</div>',
                    unsafe_allow_html=True)

        pd_means   = cache_df[cache_df["label"]==1][FEATURE_COLS].mean()
        ctrl_means = cache_df[cache_df["label"]==0][FEATURE_COLS].mean()
        diff_df = pd.DataFrame({
            "Feature":  FEATURE_COLS,
            "PD Mean":  pd_means.values,
            "Control Mean": ctrl_means.values,
            "PD - Control": (pd_means - ctrl_means).values,
        }).sort_values("PD - Control", ascending=False)

        fig_bar = px.bar(
            diff_df, x="PD - Control", y="Feature",
            orientation="h",
            color="PD - Control",
            color_continuous_scale=["#00cc66", "#161b22", "#ff6b6b"],
            color_continuous_midpoint=0,
            template="plotly_dark",
            title="Feature mean difference (PD minus Control) — positive = higher in PD",
        )
        fig_bar.update_layout(
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font=dict(color="#e6edf3", family="Inter"),
            height=560,
            margin=dict(l=200, t=50, b=20, r=20),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="diff_bar")

        # Raw data table (togglable)
        with st.expander("📋 Raw feature table"):
            show_df = cache_df.copy()
            show_df["Group"] = show_df["label"].map(
                {1: "Parkinson's", 0: "Control"}
            )
            cols_show = ["subject_id", "Group"] + FEATURE_COLS
            st.dataframe(show_df[cols_show], use_container_width=True)
