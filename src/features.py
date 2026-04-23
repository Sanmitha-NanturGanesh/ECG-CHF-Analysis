import numpy as np

def compute_hrv_features(rr_intervals):
    rr_intervals = np.array(rr_intervals)

    mean_rr = np.mean(rr_intervals)
    std_rr = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
    pnn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals)
    var_rr = np.var(rr_intervals)

    return {
        "mean_rr": mean_rr,
        "std_rr": std_rr,
        "rmssd": rmssd,
        "pnn50": pnn50,
        "var_rr": var_rr
    }
