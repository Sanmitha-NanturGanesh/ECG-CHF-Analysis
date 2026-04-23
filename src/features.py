import numpy as np
import pandas as pd


def compute_hrv_features(rr_intervals_ms):
    """
    Compute HRV features from RR intervals in milliseconds.

    Parameters:
        rr_intervals_ms (array-like): RR intervals in milliseconds

    Returns:
        dict: HRV feature dictionary
    """
    rr = np.asarray(rr_intervals_ms, dtype=float)

    if rr.ndim != 1:
        raise ValueError("RR intervals must be a 1D array.")

    if len(rr) < 3:
        raise ValueError("At least 3 RR intervals are required to compute HRV features.")

    diff_rr = np.diff(rr)

    mean_rr = np.mean(rr)
    std_rr = np.std(rr, ddof=1)
    var_rr = np.var(rr, ddof=1)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    pnn50 = (np.sum(np.abs(diff_rr) > 50) / len(diff_rr)) * 100

    return {
        "mean_rr_ms": round(mean_rr, 3),
        "std_rr_ms": round(std_rr, 3),
        "var_rr_ms2": round(var_rr, 3),
        "rmssd_ms": round(rmssd, 3),
        "pnn50_pct": round(pnn50, 3)
    }


def build_feature_table(rr_interval_dict):
    """
    Build a DataFrame of HRV features for multiple ECG records.

    Parameters:
        rr_interval_dict (dict): {record_name: rr_intervals_ms}

    Returns:
        pd.DataFrame
    """
    rows = []

    for record_name, rr_intervals in rr_interval_dict.items():
        try:
            features = compute_hrv_features(rr_intervals)
            features["record"] = record_name
            rows.append(features)
        except ValueError:
            continue

    df = pd.DataFrame(rows)

    if not df.empty:
        cols = ["record", "mean_rr_ms", "std_rr_ms", "var_rr_ms2", "rmssd_ms", "pnn50_pct"]
        df = df[cols]

    return df


def normalize_features(df, feature_columns):
    """
    Normalize selected feature columns using z-score scaling.

    Parameters:
        df (pd.DataFrame): Input feature table
        feature_columns (list): Columns to normalize

    Returns:
        pd.DataFrame: Copy with normalized feature columns
    """
    df_norm = df.copy()

    for col in feature_columns:
        mean = df_norm[col].mean()
        std = df_norm[col].std(ddof=1)

        if std == 0:
            df_norm[col] = 0
        else:
            df_norm[col] = (df_norm[col] - mean) / std

    return df_norm
