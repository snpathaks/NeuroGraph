"""
feature_extraction.py — NeuroGraph
===================================
Extracts a rich, hand-crafted feature vector from a single preprocessed
stroke DataFrame (output of utils.prepare_subject_stroke).

Novel features implemented
--------------------------
1. Velocity & speed statistics        — mean/std/max of instantaneous speed
2. Acceleration statistics            — mean/std/max of |acceleration|
3. Jerk statistics                    — rate of change of acceleration
                                        (key smoothness measure for tremor)
4. Tremor frequency (FFT)             — dominant tremor band (3-12 Hz) power
                                        ratio in x-velocity and y-velocity
5. Pressure-velocity coupling         — Pearson r between speed and pressure
                                        (PD patients tend to decouple these)
6. Adaptive segment weighting         — stroke split into N equal segments;
                                        per-segment jerk variance weighted by
                                        pen-contact density (button_status)
7. Directional entropy                — Shannon entropy of movement direction
                                        histogram (PD -> more chaotic direction)
8. Pen-lift ratio                     — fraction of samples where pen is lifted
                                        (button_status == 0)

Usage
-----
    from feature_extraction import extract_features, batch_extract

    # Single file
    fv = extract_features("data/hw_dataset/control/C_0010.txt")
    print(fv)

    # Whole dataset -> DataFrame ready for model training
    df = batch_extract("data/hw_dataset")
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from utils import prepare_subject_stroke, load_stroke_file

# ---- Constants ---------------------------------------------------------------
N_RESAMPLE    = 500      # resampling target (must match utils default)
N_SEGMENTS    = 5        # number of adaptive segments
TREMOR_LO_HZ  = 3.0     # lower bound of Parkinson tremor band
TREMOR_HI_HZ  = 12.0    # upper bound of Parkinson tremor band
N_DIR_BINS    = 16       # histogram bins for directional entropy


# ---- Low-level signal helpers ------------------------------------------------

def _safe_diff(arr: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Finite-difference derivative; avoids division by near-zero dt."""
    dt_safe = np.where(np.abs(dt) < 1e-9, 1e-9, dt)
    return np.diff(arr) / dt_safe


def _stat(arr: np.ndarray, prefix: str) -> dict:
    """Return mean, std, max, IQR of |arr| with a key prefix."""
    a = np.abs(arr)
    return {
        f"{prefix}_mean": float(np.mean(a)),
        f"{prefix}_std":  float(np.std(a)),
        f"{prefix}_max":  float(np.max(a)) if len(a) else 0.0,
        f"{prefix}_iqr":  float(np.percentile(a, 75) - np.percentile(a, 25)),
    }


# ---- Feature 1, 2, 3: Velocity, Acceleration, Jerk --------------------------

def _velocity_acceleration_features(df: pd.DataFrame) -> dict:
    """
    Computes per-sample velocity (dx/dt, dy/dt) and speed, then acceleration
    and jerk from those signals.

    Returns dict with:
        speed_{mean,std,max,iqr}
        accel_{mean,std,max,iqr}
        jerk_{mean,std,max,iqr}
    Plus hidden keys _vx, _vy, _speed, _dt for reuse downstream.
    """
    t  = df["timestamp"].values * 1e-3    # convert ms -> seconds
    x  = df["x"].values
    y  = df["y"].values

    dt = np.diff(t)                      # shape (N-1,), units: seconds

    vx = _safe_diff(x, dt)
    vy = _safe_diff(y, dt)
    speed = np.sqrt(vx**2 + vy**2)      # instantaneous speed (N-1,)

    ax   = _safe_diff(vx, dt[:-1])      # acceleration components (N-2,)
    ay   = _safe_diff(vy, dt[:-1])
    accel = np.sqrt(ax**2 + ay**2)

    jx  = _safe_diff(ax, dt[:-2])       # jerk components (N-3,)
    jy  = _safe_diff(ay, dt[:-2])
    jerk = np.sqrt(jx**2 + jy**2)

    feats = {}
    feats.update(_stat(speed, "speed"))
    feats.update(_stat(accel, "accel"))
    feats.update(_stat(jerk,  "jerk"))

    # Store intermediates for reuse in tremor + coupling features
    feats["_vx"]    = vx
    feats["_vy"]    = vy
    feats["_speed"] = speed
    feats["_dt"]    = dt

    return feats


# ---- Feature 4: Tremor Frequency via FFT ------------------------------------

def _tremor_features(vx: np.ndarray, vy: np.ndarray,
                     dt: np.ndarray) -> dict:
    """
    Estimate dominant tremor power in the 3-12 Hz band (classic PD tremor range)
    relative to total spectral power, separately for vx and vy.

    Uses Welch's method for a more stable PSD estimate than a raw FFT.
    """
    # Average sample rate from dt (already in seconds after ms->s conversion)
    fs = 1.0 / np.mean(dt)               # samples per second

    feats = {}
    for name, sig in [("vx", vx), ("vy", vy)]:
        try:
            freqs, psd = sp_signal.welch(
                sig, fs=fs,
                nperseg=min(len(sig), 64),
                scaling="density",
            )
            total_power  = np.sum(psd) + 1e-12
            tremor_mask  = (freqs >= TREMOR_LO_HZ) & (freqs <= TREMOR_HI_HZ)
            tremor_power = np.sum(psd[tremor_mask])
            feats[f"tremor_ratio_{name}"]  = float(tremor_power / total_power)
            # Peak frequency in tremor band
            if tremor_mask.any():
                peak_idx = np.argmax(psd[tremor_mask])
                feats[f"tremor_peak_hz_{name}"] = float(freqs[tremor_mask][peak_idx])
            else:
                feats[f"tremor_peak_hz_{name}"] = 0.0
        except Exception:
            feats[f"tremor_ratio_{name}"]   = 0.0
            feats[f"tremor_peak_hz_{name}"] = 0.0

    # Combined tremor ratio (mean of both axes)
    feats["tremor_ratio_combined"] = (
        feats["tremor_ratio_vx"] + feats["tremor_ratio_vy"]
    ) / 2.0

    return feats


# ---- Feature 5: Pressure-Velocity Coupling ----------------------------------

def _pressure_velocity_coupling(df: pd.DataFrame,
                                 speed: np.ndarray) -> dict:
    """
    Computes Pearson correlation between instantaneous speed and pressure.

    Healthy writers: press harder when moving slower (natural writing rhythm).
    PD writers:      coupling breaks down -> lower or near-zero correlation.
    """
    pressure = df["pressure"].values
    # Align lengths (speed is N-1 from diff)
    p_trim = pressure[:-1]
    s_trim = speed

    if len(p_trim) < 2 or np.std(p_trim) < 1e-9 or np.std(s_trim) < 1e-9:
        return {"pressure_speed_corr": 0.0}

    corr = float(np.corrcoef(p_trim, s_trim)[0, 1])
    return {"pressure_speed_corr": corr}


# ---- Feature 6: Adaptive Segment Weighting ----------------------------------

def _adaptive_segment_features(df: pd.DataFrame) -> dict:
    """
    Splits the stroke into N_SEGMENTS equal parts and computes:
      - per-segment jerk variance (high in PD segments)
      - segment weight = pen-contact fraction (button_status > 0)

    Returns weighted jerk variance across segments and per-segment stats.
    """
    n = len(df)
    seg_size = n // N_SEGMENTS
    if seg_size < 4:
        return {f"seg{i}_weighted_jerk_var": 0.0 for i in range(N_SEGMENTS)}

    feats = {}
    weighted_jerk_vars = []

    for i in range(N_SEGMENTS):
        seg = df.iloc[i * seg_size: (i + 1) * seg_size]

        t  = seg["timestamp"].values * 1e-3   # ms -> seconds
        x  = seg["x"].values
        y  = seg["y"].values
        dt = np.diff(t)

        vx = _safe_diff(x, dt)
        vy = _safe_diff(y, dt)

        if len(dt) < 3:
            jerk_var = 0.0
            weight   = 0.0
        else:
            ax = _safe_diff(vx, dt[:-1])
            ay = _safe_diff(vy, dt[:-1])
            jx = _safe_diff(ax, dt[:-2])
            jy = _safe_diff(ay, dt[:-2])
            jerk = np.sqrt(jx**2 + jy**2)
            jerk_var = float(np.var(jerk))

            # Weight = fraction of time pen is in contact (button_status > 0)
            if "button_status" in seg.columns:
                weight = float((seg["button_status"].values > 0).mean())
            else:
                weight = 1.0

        feats[f"seg{i}_jerk_var"] = jerk_var
        feats[f"seg{i}_weight"]   = weight
        weighted_jerk_vars.append(jerk_var * (weight + 1e-6))

    # Aggregate: weighted mean jerk variance across segments
    total_weight = sum(feats.get(f"seg{i}_weight", 1.0) for i in range(N_SEGMENTS))
    feats["weighted_jerk_var_mean"] = (
        sum(weighted_jerk_vars) / (total_weight + 1e-9)
    )

    return feats


# ---- Feature 7: Directional Entropy -----------------------------------------

def _directional_entropy(df: pd.DataFrame) -> dict:
    """
    Computes Shannon entropy of the movement direction histogram.

    Healthy spiral: directions follow smooth arc -> low entropy.
    PD tremor:      erratic direction changes -> high entropy.
    """
    x = df["x"].values
    y = df["y"].values
    dx = np.diff(x)
    dy = np.diff(y)

    angles = np.arctan2(dy, dx)                          # [-pi, pi]
    counts, _ = np.histogram(angles, bins=N_DIR_BINS,
                              range=(-np.pi, np.pi))
    probs = counts / (counts.sum() + 1e-12)
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

    return {"directional_entropy": entropy}


# ---- Feature 8: Pen-lift ratio ----------------------------------------------

def _pen_lift_ratio(raw_df: pd.DataFrame) -> dict:
    """
    Fraction of raw samples where the pen is lifted (button_status == 0).
    PD patients tend to lift and re-place the pen more often.
    """
    if "button_status" not in raw_df.columns:
        return {"pen_lift_ratio": 0.0}
    lifted = (raw_df["button_status"].values == 0).mean()
    return {"pen_lift_ratio": float(lifted)}


# ---- Button-status re-attachment helper -------------------------------------

def _interpolate_button_status(raw_df: pd.DataFrame,
                                proc_df: pd.DataFrame) -> np.ndarray:
    """
    Map raw button_status back onto resampled timestamps by nearest-neighbor
    lookup on original timestamps.
    """
    raw_t   = raw_df["timestamp"].values.astype(float)
    raw_btn = raw_df["button_status"].values

    # Normalize raw timestamps to same 0-start as proc_df
    raw_t_norm = raw_t - raw_t.min()
    proc_t     = proc_df["timestamp"].values

    # Nearest neighbor
    indices = np.searchsorted(raw_t_norm, proc_t, side="left")
    indices = np.clip(indices, 0, len(raw_btn) - 1)
    return raw_btn[indices]


# ---- Master extractor -------------------------------------------------------

def extract_features(filepath: str,
                     test_id_filter: int = None,
                     n_points: int = N_RESAMPLE) -> dict:
    """
    Full feature extraction pipeline for ONE stroke file.

    Parameters
    ----------
    filepath       : path to a raw .txt stroke file
    test_id_filter : if not None, keep only rows with this test_id before
                     processing (e.g. 0 = Static Spiral, 1 = Dynamic Spiral)
    n_points       : number of resampled points (default 500)

    Returns
    -------
    dict of {feature_name: float} ready for pd.DataFrame row insertion.
    """
    # Load raw file (needed for pen_lift_ratio before resampling)
    raw_df = load_stroke_file(filepath)

    if test_id_filter is not None:
        raw_df = raw_df[raw_df["test_id"] == test_id_filter].reset_index(drop=True)
        if raw_df.empty:
            raise ValueError(
                f"No rows with test_id=={test_id_filter} in {filepath}"
            )

    # Preprocessed (normalized + resampled) stroke
    proc_df = prepare_subject_stroke(
        filepath, n_points=n_points, test_id_filter=test_id_filter
    )

    feats: dict = {"filepath": filepath}

    # 1, 2, 3 — velocity, acceleration, jerk
    va_feats = _velocity_acceleration_features(proc_df)
    vx    = va_feats.pop("_vx")
    vy    = va_feats.pop("_vy")
    speed = va_feats.pop("_speed")
    dt    = va_feats.pop("_dt")
    feats.update(va_feats)

    # 4 — tremor frequency
    feats.update(_tremor_features(vx, vy, dt))

    # 5 — pressure-velocity coupling
    feats.update(_pressure_velocity_coupling(proc_df, speed))

    # 6 — adaptive segment weighting (on normalized stroke)
    seg_df = proc_df.copy()
    if "button_status" in raw_df.columns:
        seg_df["button_status"] = _interpolate_button_status(raw_df, proc_df)
    feats.update(_adaptive_segment_features(seg_df))

    # 7 — directional entropy
    feats.update(_directional_entropy(proc_df))

    # 8 — pen-lift ratio (from raw, before resampling)
    feats.update(_pen_lift_ratio(raw_df))

    return feats


# ---- Batch extractor --------------------------------------------------------

def batch_extract(root_dir: str,
                  test_id_filter: int = None,
                  n_points: int = N_RESAMPLE) -> pd.DataFrame:
    """
    Run extract_features on every subject in root_dir and return a DataFrame
    with one row per subject and a 'label' column (1 = PD, 0 = Control).

    root_dir should contain parkinson/ and control/ sub-folders.
    """
    label_map = {"parkinson": 1, "control": 0}
    rows = []

    for label_folder, label_val in label_map.items():
        folder = os.path.join(root_dir, label_folder)
        if not os.path.isdir(folder):
            print(f"Warning: Skipping missing folder: {folder}")
            continue

        files = sorted(f for f in os.listdir(folder) if f.endswith(".txt"))
        print(f"\n-- {label_folder.upper()} ({len(files)} files) --")

        for fname in files:
            fpath = os.path.join(folder, fname)
            try:
                row = extract_features(fpath,
                                       test_id_filter=test_id_filter,
                                       n_points=n_points)
                row["subject_id"] = fname.replace(".txt", "")
                row["label"]      = label_val
                rows.append(row)
                print(f"  OK {fname}")
            except Exception as e:
                print(f"  FAIL {fname}: {e}")

    if not rows:
        raise RuntimeError(f"No features extracted from {root_dir}")

    df = pd.DataFrame(rows)

    # Move metadata cols to front
    front = ["subject_id", "label", "filepath"]
    other = [c for c in df.columns if c not in front]
    return df[front + other]


# ---- Sanity-check printer ---------------------------------------------------

def _print_comparison(ctrl_feats: dict, pd_feats: dict) -> None:
    """Print a side-by-side table: one healthy vs one PD sample."""
    skip = {"filepath", "subject_id", "label"}
    numeric_keys = [k for k in ctrl_feats if k not in skip
                    and isinstance(ctrl_feats.get(k), float)]

    print(f"\n{'Feature':<38} {'Control':>12} {'Parkinson':>12}  {'Direction':>9}")
    print("-" * 75)
    for k in sorted(numeric_keys):
        cv = ctrl_feats.get(k, float("nan"))
        pv = pd_feats.get(k, float("nan"))
        marker = "  UP (PD)" if pv > cv else ("  DOWN (PD)" if pv < cv else "")
        print(f"  {k:<36} {cv:>12.6f} {pv:>12.6f}{marker}")


# ---- Main: sanity-check on 4 files ------------------------------------------

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    CTRL_FILES = [
        "data/hw_dataset/control/C_0010.txt",
        "data/hw_dataset/control/C_0001.txt",
    ]
    PD_FILES = [
        "data/hw_dataset/parkinson/P_02100001.txt",
        "data/hw_dataset/parkinson/P_05060003.txt",
    ]

    print("=" * 75)
    print("NeuroGraph -- Feature Extraction Sanity Check")
    print("=" * 75)

    ctrl_results = []
    pd_results   = []

    for fpath in CTRL_FILES:
        if not os.path.exists(fpath):
            print(f"File not found: {fpath}")
            continue
        print(f"\n[CONTROL] {os.path.basename(fpath)}")
        try:
            feats = extract_features(fpath)
            ctrl_results.append(feats)
            print(f"  speed_mean        = {feats['speed_mean']:.6f}")
            print(f"  jerk_mean         = {feats['jerk_mean']:.6f}")
            print(f"  jerk_std          = {feats['jerk_std']:.6f}")
            print(f"  tremor_ratio_comb = {feats['tremor_ratio_combined']:.6f}")
            print(f"  directional_entr  = {feats['directional_entropy']:.6f}")
            print(f"  pen_lift_ratio    = {feats['pen_lift_ratio']:.6f}")
            print(f"  pressure_spd_corr = {feats['pressure_speed_corr']:.6f}")
            print(f"  weighted_jerk_var = {feats['weighted_jerk_var_mean']:.6f}")
        except Exception as e:
            print(f"  ERROR: {e}")

    for fpath in PD_FILES:
        if not os.path.exists(fpath):
            print(f"File not found: {fpath}")
            continue
        print(f"\n[PARKINSON] {os.path.basename(fpath)}")
        try:
            feats = extract_features(fpath)
            pd_results.append(feats)
            print(f"  speed_mean        = {feats['speed_mean']:.6f}")
            print(f"  jerk_mean         = {feats['jerk_mean']:.6f}")
            print(f"  jerk_std          = {feats['jerk_std']:.6f}")
            print(f"  tremor_ratio_comb = {feats['tremor_ratio_combined']:.6f}")
            print(f"  directional_entr  = {feats['directional_entropy']:.6f}")
            print(f"  pen_lift_ratio    = {feats['pen_lift_ratio']:.6f}")
            print(f"  pressure_spd_corr = {feats['pressure_speed_corr']:.6f}")
            print(f"  weighted_jerk_var = {feats['weighted_jerk_var_mean']:.6f}")
        except Exception as e:
            print(f"  ERROR: {e}")

    if ctrl_results and pd_results:
        print("\n" + "=" * 75)
        print("Side-by-side comparison (C_0010 vs P_02100001):")
        _print_comparison(ctrl_results[0], pd_results[0])
