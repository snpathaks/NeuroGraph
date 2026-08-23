"""
model.py — NeuroGraph
======================
End-to-end ML pipeline:

    1. Batch feature extraction (feature_extraction.batch_extract)
    2. Train / test split (stratified)
    3. Train Random Forest  -> models/neurograph_rf.pkl
    4. Train XGBoost        -> models/neurograph_xgb.pkl
    5. Evaluate both: accuracy, F1, ROC-AUC, confusion matrix
    6. Feature importance plot (saved to models/feature_importance.png)

Usage
-----
    python model.py                   # train on full dataset
    python model.py --skip-extract    # reload cached features CSV, skip batch_extract
"""

import os
import sys
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — safe on any machine
import matplotlib.pyplot as plt

from sklearn.ensemble      import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics       import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed — skipping XGBoost training.")

from feature_extraction import batch_extract

# ---- Paths ------------------------------------------------------------------
DATA_DIR         = "data/hw_dataset"
MODELS_DIR       = "models"
FEATURES_CACHE   = os.path.join(MODELS_DIR, "features_cache.csv")
RF_MODEL_PATH    = os.path.join(MODELS_DIR, "neurograph_rf.pkl")
XGB_MODEL_PATH   = os.path.join(MODELS_DIR, "neurograph_xgb.pkl")
FI_PLOT_PATH     = os.path.join(MODELS_DIR, "feature_importance.png")
ROC_PLOT_PATH    = os.path.join(MODELS_DIR, "roc_curve.png")

# ---- Feature columns to drop before training --------------------------------
# These are metadata / string cols, not numeric features
META_COLS = ["subject_id", "label", "filepath"]

# Segment weight columns are useful but correlated; keep them in by default
DROP_COLS: list[str] = []   # add col names here to exclude from training


# =============================================================================
# Step 1 — Feature extraction
# =============================================================================

def load_or_extract(skip_extract: bool = False) -> pd.DataFrame:
    """
    Either re-run batch_extract on the whole dataset, or reload a cached CSV.
    The cache is always written so future runs can use --skip-extract.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    if skip_extract and os.path.exists(FEATURES_CACHE):
        print(f"Loading cached features from {FEATURES_CACHE}")
        df = pd.read_csv(FEATURES_CACHE)
        print(f"  {len(df)} subjects loaded, {df['label'].sum()} PD / "
              f"{(df['label'] == 0).sum()} Control")
        return df

    print("=" * 65)
    print("Step 1: Batch feature extraction")
    print("=" * 65)
    df = batch_extract(DATA_DIR)

    df.to_csv(FEATURES_CACHE, index=False)
    print(f"\nFeatures cached to {FEATURES_CACHE}")
    print(f"Dataset: {len(df)} subjects | {df['label'].sum()} PD | "
          f"{(df['label'] == 0).sum()} Control")
    return df


# =============================================================================
# Step 2 — Prepare X / y
# =============================================================================

def prepare_xy(df: pd.DataFrame):
    """
    Returns X (feature matrix), y (labels), and feature_names list.
    Drops metadata and any columns in DROP_COLS.
    Fills NaN with column median.
    """
    drop = set(META_COLS + DROP_COLS)
    feature_cols = [c for c in df.columns if c not in drop]

    X = df[feature_cols].copy()

    # Fill any NaN values with column medians
    nan_counts = X.isna().sum()
    if nan_counts.any():
        print(f"\nFilling NaN values in: {list(nan_counts[nan_counts > 0].index)}")
        X = X.fillna(X.median(numeric_only=True))

    # Replace infinities
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    y = df["label"].values
    return X.values, y, feature_cols


# =============================================================================
# Step 3 — Evaluation helper
# =============================================================================

def evaluate(name: str, model, X_test: np.ndarray,
             y_test: np.ndarray, cv_scores: np.ndarray) -> dict:
    """
    Print and return a dict of evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else float("nan")
    cm      = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*65}")
    print(f"  {name} — Test-set results")
    print(f"{'='*65}")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  F1 Score   : {f1:.4f}")
    print(f"  ROC-AUC    : {roc_auc:.4f}")
    print(f"  CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}  "
          f"(5-fold stratified)")
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"             Pred:Control  Pred:PD")
    print(f"  Act:Control    {cm[0,0]:>5}       {cm[0,1]:>5}")
    print(f"  Act:PD         {cm[1,0]:>5}       {cm[1,1]:>5}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Control", "Parkinson"],
                                zero_division=0))

    return {
        "name": name, "accuracy": acc, "f1": f1, "roc_auc": roc_auc,
        "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
        "y_pred": y_pred, "y_prob": y_prob,
    }


# =============================================================================
# Step 4 — Feature importance plot
# =============================================================================

def plot_feature_importance(rf_model, feature_names: list,
                             top_n: int = 20) -> None:
    """Save a horizontal bar chart of the top-N RF feature importances."""
    # Extract from pipeline if wrapped
    clf = rf_model.named_steps["clf"] if hasattr(rf_model, "named_steps") else rf_model

    importances = clf.feature_importances_
    idx = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, top_n))
    ax.barh(range(top_n), importances[idx], color=colors, edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx], fontsize=9)
    ax.set_xlabel("Gini Importance", fontsize=10)
    ax.set_title(f"NeuroGraph — Top {top_n} Feature Importances (Random Forest)",
                 fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(FI_PLOT_PATH, dpi=150)
    plt.close()
    print(f"\nFeature importance plot saved to {FI_PLOT_PATH}")


# =============================================================================
# Step 5 — ROC curve plot
# =============================================================================

def plot_roc(results: list) -> None:
    """Save an overlaid ROC curve for all trained models."""
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    for res, color in zip(results, colors):
        fpr, tpr, _ = roc_curve(res["y_test"], res["y_prob"])
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{res['name']}  (AUC = {res['roc_auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("NeuroGraph — ROC Curves", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(ROC_PLOT_PATH, dpi=150)
    plt.close()
    print(f"ROC curve plot saved to {ROC_PLOT_PATH}")


# =============================================================================
# Step 6 — Save model
# =============================================================================

def save_model(model, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


# =============================================================================
# Main training pipeline
# =============================================================================

def train(skip_extract: bool = False) -> None:
    # ------------------------------------------------------------------
    # 1. Load features
    # ------------------------------------------------------------------
    df = load_or_extract(skip_extract)

    if len(df) < 4:
        print("ERROR: Need at least 4 subjects to train. Check your data path.")
        sys.exit(1)

    X, y, feature_names = prepare_xy(df)
    n_samples, n_features = X.shape
    n_pd      = int(y.sum())
    n_ctrl    = int((y == 0).sum())

    print(f"\n{'='*65}")
    print(f"Step 2: Dataset summary")
    print(f"{'='*65}")
    print(f"  Total subjects : {n_samples}")
    print(f"  PD             : {n_pd}")
    print(f"  Control        : {n_ctrl}")
    print(f"  Features       : {n_features}")

    # ------------------------------------------------------------------
    # 2. Train / test split (stratified, 80/20)
    # ------------------------------------------------------------------
    # With only 40 subjects a 20% hold-out is ~8 samples; use a small
    # test size but keep stratification to ensure both classes appear.
    test_size = 0.20 if n_samples >= 20 else 0.25
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")
    print(f"Train PD: {y_train.sum()}  |  Test PD: {y_test.sum()}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_results = []

    # ------------------------------------------------------------------
    # 3. Random Forest
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("Step 3: Training Random Forest")
    print(f"{'='*65}")

    rf_pipeline = Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",   # handles PD/Control imbalance
            random_state=42,
            n_jobs=-1,
        ))
    ])

    rf_pipeline.fit(X_train, y_train)
    save_model(rf_pipeline, RF_MODEL_PATH)

    rf_cv = cross_val_score(rf_pipeline, X, y, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    rf_res = evaluate("Random Forest", rf_pipeline, X_test, y_test, rf_cv)
    rf_res["y_test"] = y_test
    all_results.append(rf_res)

    # Feature importance (RF only)
    plot_feature_importance(rf_pipeline, feature_names)

    # ------------------------------------------------------------------
    # 4. XGBoost (optional)
    # ------------------------------------------------------------------
    if XGBOOST_AVAILABLE:
        print(f"\n{'='*65}")
        print("Step 4: Training XGBoost")
        print(f"{'='*65}")

        scale_pos = n_ctrl / max(n_pd, 1)   # handles class imbalance

        xgb_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            ))
        ])

        xgb_pipeline.fit(X_train, y_train)
        save_model(xgb_pipeline, XGB_MODEL_PATH)

        xgb_cv = cross_val_score(xgb_pipeline, X, y, cv=cv,
                                  scoring="roc_auc", n_jobs=-1)
        xgb_res = evaluate("XGBoost", xgb_pipeline, X_test, y_test, xgb_cv)
        xgb_res["y_test"] = y_test
        all_results.append(xgb_res)

    # ------------------------------------------------------------------
    # 5. ROC curve comparison
    # ------------------------------------------------------------------
    if len(all_results) >= 1:
        # Only plot if test set has both classes
        if len(np.unique(y_test)) > 1:
            plot_roc(all_results)
        else:
            print("\nNote: Test set has only one class — ROC curve skipped.")

    # ------------------------------------------------------------------
    # 6. Summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("FINAL SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Model':<20} {'Accuracy':>9} {'F1':>8} {'ROC-AUC':>9} "
          f"{'CV AUC (mean)':>14}")
    print("  " + "-" * 62)
    for r in all_results:
        print(f"  {r['name']:<20} {r['accuracy']:>9.4f} {r['f1']:>8.4f} "
              f"{r['roc_auc']:>9.4f} {r['cv_mean']:>14.4f} +/- {r['cv_std']:.4f}")

    print(f"\nBest model by ROC-AUC: "
          f"{max(all_results, key=lambda r: r['roc_auc'])['name']}")
    print("\nDone. Models saved to models/")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NeuroGraph — train PD detection models"
    )
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="Reload cached features CSV instead of re-running batch_extract"
    )
    args = parser.parse_args()
    train(skip_extract=args.skip_extract)
