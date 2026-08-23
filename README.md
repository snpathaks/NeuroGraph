# 🧠 NeuroGraph — Parkinson's Detection via Handwriting Analysis
streamlit : https://neurograph-hrzhqmnq8qdouqmarsm3c3.streamlit.app/

> **Early Parkinson's Disease detection from digitized pen-tablet spiral drawings using handcrafted signal features and ensemble machine learning.**

---

## 📌 Project Overview

When someone has a tremor (like early Parkinson's), their hand shakes slightly while writing — you can't always see it, but a **digital pen tablet** can measure it: how much pressure changes, how fast the pen moves, how jittery the strokes are. NeuroGraph turns those measurements into a rich feature vector and trains ML models to classify **Parkinson's Disease (PD)** vs **Healthy Control**.

The novel contribution is the **8-group handcrafted feature set** — not the algorithm itself.

---

## 🗂️ Dataset

| Property | Details |
|---|---|
| **Source** | UCI — *Parkinson Disease Spiral Drawings Using Digitized Graphics Tablet* |
| **Reference** | Isenkul et al., "Improved spiral test using digitized graphics tablet for monitoring Parkinson's disease." ICEHTM 2014 |
| **Input Format** | Space/semicolon-separated `.txt` stroke files (no header) |
| **Classes** | `parkinson/` (label = 1) · `control/` (label = 0) |
| **Test Types** | 0 = Static Spiral · 1 = Dynamic Spiral · 2 = Stability Test |
| **Resampling** | Each stroke uniformly resampled to **500 points** before feature extraction |

**Raw column order per file:**

| Column | Description |
|---|---|
| X, Y | Pen coordinates |
| Z (button_status) | Pen contact (0 = lifted, 1 = down) |
| Pressure | Axial pressure of the pen |
| GripAngle | Pen grip angle |
| Timestamp | Time in milliseconds |
| TestID | 0 / 1 / 2 |

---

## ⚙️ Classification Task

| Property | Value |
|---|---|
| **Task Type** | Binary Classification |
| **Positive Class** | Parkinson's Disease (label = 1) |
| **Negative Class** | Healthy Control (label = 0) |
| **Train / Test Split** | 80 % / 20 % (stratified) |
| **Cross-Validation** | 5-fold Stratified K-Fold |
| **Class Imbalance Handling** | `class_weight="balanced"` (RF) · `scale_pos_weight` (XGBoost) |

---

## 🔬 Feature Engineering — 8 Novel Feature Groups

All features are extracted from the preprocessed (normalized + resampled) stroke signal.

| # | Feature Group | Features Extracted | Key Insight |
|---|---|---|---|
| 1 | **Velocity & Speed** | `speed_mean`, `speed_std`, `speed_max`, `speed_iqr` | PD patients write slower with higher variance |
| 2 | **Acceleration** | `accel_mean`, `accel_std`, `accel_max`, `accel_iqr` | Erratic acceleration changes indicate motor impairment |
| 3 | **Jerk** | `jerk_mean`, `jerk_std`, `jerk_max`, `jerk_iqr` | Jerk (d³pos/dt³) is the primary smoothness proxy for tremor |
| 4 | **Tremor Frequency (FFT)** | `tremor_ratio_vx`, `tremor_ratio_vy`, `tremor_ratio_combined`, `tremor_peak_hz_vx`, `tremor_peak_hz_vy` | Welch PSD — ratio of power in **3–12 Hz** (PD tremor band) to total spectral power |
| 5 | **Pressure-Velocity Coupling** | `pressure_speed_corr` | Healthy: slower → more pressure; PD: coupling breaks down → near-zero Pearson r |
| 6 | **Adaptive Segment Weighting** | `seg0..4_jerk_var`, `seg0..4_weight`, `weighted_jerk_var_mean` | Stroke split into 5 segments; per-segment jerk variance weighted by pen-contact density |
| 7 | **Directional Entropy** | `directional_entropy` | Shannon entropy of 16-bin direction histogram; PD → chaotic angles → higher entropy |
| 8 | **Pen-Lift Ratio** | `pen_lift_ratio` | Fraction of samples where pen is lifted; PD patients re-place more often |

**Total features per subject: 31**

---

## 🤖 Models Used

### 1. Random Forest Classifier

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | None (fully grown trees) |
| `min_samples_leaf` | 1 |
| `class_weight` | `balanced` |
| `random_state` | 42 |
| Pipeline | Raw features → RF (no scaling needed) |
| Saved to | `models/neurograph_rf.pkl` |

### 2. XGBoost Classifier

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | 4 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `scale_pos_weight` | `n_control / n_pd` (dynamic) |
| `eval_metric` | `logloss` |
| `random_state` | 42 |
| Pipeline | `StandardScaler` → XGBoost |
| Saved to | `models/neurograph_xgb.pkl` |

---

## 📊 Model Performance

> Metrics computed on the **20% held-out test set** (stratified split, `random_state=42`).
> Cross-validation scores use **5-fold Stratified K-Fold** on the full dataset (ROC-AUC scoring).

### Test-Set Results

| Model | Accuracy | F1 Score | ROC-AUC | CV AUC (mean ± std) |
|---|---|---|---|---|
| **Random Forest** | **0.8750 (87.5%)** | **0.9091** | **0.9333** | **0.8533 ± 0.0980** |
| **XGBoost** | **0.6250 (62.5%)** | **0.7273** | **0.8667** | **0.8200 ± 0.1485** |

> 🏆 **Best Model:** Random Forest achieved top performance across all metrics with **0.9333 ROC-AUC** and **87.5% test accuracy**.

### Evaluation Metrics Explained

| Metric | Description |
|---|---|
| **Accuracy** | `(TP + TN) / total` — overall correct predictions |
| **F1 Score** | Harmonic mean of Precision & Recall — balances false positives/negatives |
| **ROC-AUC** | Area under the ROC curve — discrimination ability across all thresholds |
| **CV AUC** | Mean ± std of ROC-AUC across 5 folds — measures generalization stability |

### Confusion Matrix Structure

```
                Pred: Control   Pred: PD
Act: Control        TN              FP
Act: PD             FN              TP
```

---

## 🏗️ Project Structure

```
NeuroGraph/
├── app.py                   # Streamlit web application (Live Draw, Upload, Explorer)
├── model.py                 # End-to-end ML training pipeline
├── feature_extraction.py    # 8-group handcrafted feature extractor
├── utils.py                 # Raw stroke file loader, normalizer, resampler
├── dashboard.html/.css/.js  # Standalone HTML dashboard
├── data/
│   └── hw_dataset/
│       ├── parkinson/       # PD subject .txt files  (P_*.txt)
│       └── control/         # Healthy subject .txt files (C_*.txt)
└── models/
    ├── neurograph_rf.pkl    # Trained Random Forest model
    ├── neurograph_xgb.pkl   # Trained XGBoost model
    ├── features_cache.csv   # Cached feature matrix (skip re-extraction)
    ├── feature_importance.png
    └── roc_curve.png
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install numpy pandas scipy scikit-learn xgboost streamlit plotly streamlit-drawable-canvas matplotlib
```

### 2. Train the models

```bash
# Full run: extract features + train both models
python model.py

# Skip feature extraction if features_cache.csv already exists
python model.py --skip-extract
```

### 3. Launch the Streamlit app

```bash
streamlit run app.py
```

---

## 🖥️ Application Tabs

| Tab | Description |
|---|---|
| **🎨 Live Draw** | Draw a spiral on the canvas; pen data is simulated; features extracted live; prediction shown instantly |
| **📁 Upload CSV** | Upload a raw stroke `.txt` file (UCI format); full feature extraction + model prediction |
| **📊 Dataset Explorer** | Browse `features_cache.csv`; compare PD vs Control feature distributions |

---

## 🔄 ML Pipeline — Step by Step

```
Raw .txt stroke files
        │
        ▼
[utils.py]  Load → Clean → Normalize → Resample to 500 pts
        │
        ▼
[feature_extraction.py]  8 feature groups → 31-dim vector per subject
        │
        ▼
[model.py]  features_cache.csv → 80/20 stratified split
        │
        ├──▶ Random Forest  (300 trees, balanced weights) → neurograph_rf.pkl
        │
        └──▶ XGBoost  (300 trees, lr=0.05, depth=4) → neurograph_xgb.pkl
                │
                ▼
     Evaluate: Accuracy · F1 · ROC-AUC · 5-fold CV · Confusion Matrix
```

---

## 📈 Feature Importance

Top features by Random Forest Gini Importance (plot saved to `models/feature_importance.png`):

| Rank | Expected Top Features | Rationale |
|---|---|---|
| 1 | `jerk_std` / `jerk_mean` | Primary tremor signal — high in PD |
| 2 | `tremor_ratio_combined` | Direct 3–12 Hz spectral power ratio |
| 3 | `weighted_jerk_var_mean` | Segment-weighted tremor severity |
| 4 | `directional_entropy` | Erratic direction-change measure |
| 5 | `pressure_speed_corr` | Coupling breakdown in PD |
| 6 | `pen_lift_ratio` | More frequent pen lifts in PD |

---


