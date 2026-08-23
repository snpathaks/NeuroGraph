"""
utils.py — NeuroGraph
Helper functions for loading raw pen-tablet stroke files and preparing
them (cleaning, normalizing, resampling) before feature extraction.

Dataset: UCI "Parkinson Disease Spiral Drawings Using Digitized Graphics Tablet"
Source paper: Isenkul, M.E.; Sakar, B.E.; Kursun, O. 'Improved spiral test using
digitized graphics tablet for monitoring Parkinson's disease.' ICEHTM 2014.

IMPORTANT — VERIFY BEFORE TRUSTING:
The documented column order for hw_dataset/*.txt files (space-separated,
no header row) is:
    X, Y, Z, Pressure, GripAngle, Timestamp, TestID
where:
    X, Y       = pen coordinates
    Z          = button status (pen up/down, 0 or 1)
    Pressure   = axial pressure of the pen
    GripAngle  = angle of the pen grip
    Timestamp  = time in milliseconds
    TestID     = 0 = Static Spiral Test, 1 = Dynamic Spiral Test, 2 = Stability Test

I have NOT been able to verify this against a real file (network sandbox
couldn't reach archive.ics.uci.edu). Run `inspect_raw_file()` below on one
real .txt file first — check the printed shape and value ranges make sense
(X/Y should look like coordinates, Timestamp should be increasing, Pressure
should be a small positive range) before trusting downstream results. If the
columns look wrong, adjust COLUMN_NAMES below.
"""

import os
import numpy as np
import pandas as pd

# ---- Adjust this if inspect_raw_file() shows a different column order ----
COLUMN_NAMES = ["x", "y", "button_status", "pressure", "grip_angle", "timestamp", "test_id"]


def inspect_raw_file(filepath: str, n_rows: int = 10) -> pd.DataFrame:
    """
    Quick sanity check on one raw .txt file. Run this FIRST on a real file
    before trusting the rest of the pipeline.

    Prints shape, dtypes, and first n_rows so you can visually confirm the
    column order matches COLUMN_NAMES.
    """
    df = pd.read_csv(filepath, sep=";", header=None)
    print(f"File: {filepath}")
    print(f"Shape: {df.shape}  (rows, columns)")
    print(f"\nFirst {n_rows} rows (raw, no column names applied yet):")
    print(df.head(n_rows))
    print(f"\nColumn-wise min/max (to sanity check which column is which):")
    print(df.describe().T[["min", "max"]])

    if df.shape[1] != len(COLUMN_NAMES):
        print(
            f"\n⚠️  WARNING: file has {df.shape[1]} columns but COLUMN_NAMES "
            f"has {len(COLUMN_NAMES)} entries. Update COLUMN_NAMES in utils.py."
        )
    return df


def load_stroke_file(filepath: str) -> pd.DataFrame:
    """
    Load a single raw stroke file into a clean, labeled DataFrame.

    Returns a DataFrame with columns matching COLUMN_NAMES, sorted by timestamp.
    """
    df = pd.read_csv(filepath, sep=";", header=None)

    if df.shape[1] != len(COLUMN_NAMES):
        raise ValueError(
            f"{filepath}: expected {len(COLUMN_NAMES)} columns, got {df.shape[1]}. "
            f"Run inspect_raw_file() on this file and fix COLUMN_NAMES."
        )

    df.columns = COLUMN_NAMES
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_dataset(root_dir: str) -> pd.DataFrame:
    """
    Walk the dataset root directory (e.g. data/uci_spiral_data/hw_dataset),
    load every subject file, and return one combined DataFrame with an added
    'subject_id', 'label' (1 = Parkinson's, 0 = Healthy), and 'filepath' column.

    Expects the standard UCI folder layout:
        hw_dataset/parkinson/*.txt   -> label 1
        hw_dataset/control/*.txt     -> label 0
    """
    records = []

    label_map = {"parkinson": 1, "control": 0}

    for label_folder, label_value in label_map.items():
        folder_path = os.path.join(root_dir, label_folder)
        if not os.path.isdir(folder_path):
            print(f"⚠️  Folder not found, skipping: {folder_path}")
            continue

        for fname in sorted(os.listdir(folder_path)):
            if not fname.endswith(".txt"):
                continue
            filepath = os.path.join(folder_path, fname)
            try:
                stroke_df = load_stroke_file(filepath)
            except Exception as e:
                print(f"⚠️  Skipping {filepath}: {e}")
                continue

            stroke_df["subject_id"] = fname.replace(".txt", "")
            stroke_df["label"] = label_value
            stroke_df["filepath"] = filepath
            records.append(stroke_df)

    if not records:
        raise FileNotFoundError(
            f"No valid stroke files found under {root_dir}. "
            f"Check that the UCI zip was extracted correctly."
        )

    combined = pd.concat(records, ignore_index=True)
    print(f"Loaded {combined['subject_id'].nunique()} subjects "
          f"({(combined.groupby('subject_id')['label'].first() == 1).sum()} Parkinson's, "
          f"{(combined.groupby('subject_id')['label'].first() == 0).sum()} Healthy)")
    return combined


def normalize_stroke(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a single subject's stroke so different hand sizes / tablet
    positioning don't distort feature values.

    - Centers X, Y around their own mean (removes absolute tablet position)
    - Scales X, Y to unit range (removes absolute drawing size differences)
    - Converts timestamp to start at 0 (removes absolute recording start time)
    - Scales pressure to 0-1 range
    """
    df = df.copy()

    for axis in ["x", "y"]:
        df[axis] = df[axis] - df[axis].mean()
        span = df[axis].max() - df[axis].min()
        if span > 0:
            df[axis] = df[axis] / span

    df["timestamp"] = df["timestamp"] - df["timestamp"].min()

    p_span = df["pressure"].max() - df["pressure"].min()
    if p_span > 0:
        df["pressure"] = (df["pressure"] - df["pressure"].min()) / p_span

    return df


def resample_stroke(df: pd.DataFrame, n_points: int = 500) -> pd.DataFrame:
    """
    Resample a stroke to a fixed number of points using linear interpolation
    over timestamp. This makes strokes of different original lengths
    comparable when extracting features later.
    """
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp")

    if len(df) < 2:
        raise ValueError("Stroke too short to resample (needs at least 2 points).")

    t_original = df["timestamp"].values
    t_new = np.linspace(t_original.min(), t_original.max(), n_points)

    resampled = {"timestamp": t_new}
    for col in ["x", "y", "pressure"]:
        resampled[col] = np.interp(t_new, t_original, df[col].values)

    return pd.DataFrame(resampled)


def prepare_subject_stroke(filepath: str, n_points: int = 500, test_id_filter: int = None) -> pd.DataFrame:
    """
    Convenience wrapper: load -> (optionally filter by test type) -> normalize -> resample.
    Use this as the single entry point when building the feature extraction pipeline.

    test_id_filter: if set (e.g. 0 for Static Spiral Test only), keeps only
    rows matching that test_id before processing. Recommended, since mixing
    test types (static/dynamic/stability) in one feature vector may not be
    meaningful — decide this once you've confirmed the test_id column.
    """
    df = load_stroke_file(filepath)

    if test_id_filter is not None:
        df = df[df["test_id"] == test_id_filter].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No rows with test_id == {test_id_filter} in {filepath}")

    df = normalize_stroke(df)
    df = resample_stroke(df, n_points=n_points)
    return df


if __name__ == "__main__":
    # Quick manual test — point this at ONE real file from your downloaded dataset
    # to confirm everything works before moving to feature_extraction.py
    sample_file = "data/hw_dataset/control/C_0010.txt"

    if os.path.exists(sample_file):
        print("=== Step 1: Inspect raw file ===")
        inspect_raw_file(sample_file)

        print("\n=== Step 2: Load + normalize + resample ===")
        processed = prepare_subject_stroke(sample_file, n_points=500)
        print(processed.head())
        print(f"\nProcessed shape: {processed.shape}")
    else:
        print(f"Sample file not found at {sample_file}. "
              f"Update the path to point at a real file from your dataset.")