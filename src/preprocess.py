import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def bandpass_filter(signal, lowcut=0.5, highcut=50.0, fs=250, order=2):
    """
    Apply a Butterworth bandpass filter to an ECG signal.

    Parameters:
        signal (array-like): Raw ECG signal
        lowcut (float): Low cutoff frequency in Hz
        highcut (float): High cutoff frequency in Hz
        fs (int): Sampling frequency in Hz
        order (int): Filter order

    Returns:
        np.ndarray: Filtered ECG signal
    """
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError("Signal must be a 1D array.")

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    if not 0 < low < high < 1:
        raise ValueError("Invalid cutoff frequencies. Ensure 0 < lowcut < highcut < fs/2.")

    b, a = butter(order, [low, high], btype="band")
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def detect_r_peaks(filtered_signal, fs=250, min_distance_sec=0.4, prominence_factor=0.5):
    """
    Detect approximate R-peaks from a filtered ECG signal.

    Parameters:
        filtered_signal (array-like): Filtered ECG signal
        fs (int): Sampling frequency
        min_distance_sec (float): Minimum time between peaks
        prominence_factor (float): Peak prominence multiplier

    Returns:
        np.ndarray: Indices of detected R-peaks
    """
    filtered_signal = np.asarray(filtered_signal, dtype=float)

    if filtered_signal.ndim != 1:
        raise ValueError("Filtered signal must be a 1D array.")

    min_distance = int(min_distance_sec * fs)
    prominence = prominence_factor * np.std(filtered_signal)

    peaks, _ = find_peaks(filtered_signal, distance=min_distance, prominence=prominence)
    return peaks


def compute_rr_intervals(r_peaks, fs=250, unit="ms"):
    """
    Compute RR intervals from R-peak indices.

    Parameters:
        r_peaks (array-like): Indices of detected R-peaks
        fs (int): Sampling frequency
        unit (str): 'ms' or 'sec'

    Returns:
        np.ndarray: RR intervals
    """
    r_peaks = np.asarray(r_peaks)

    if len(r_peaks) < 2:
        return np.array([])

    rr_intervals_sec = np.diff(r_peaks) / fs

    if unit == "ms":
        return rr_intervals_sec * 1000
    if unit == "sec":
        return rr_intervals_sec

    raise ValueError("unit must be either 'ms' or 'sec'")


def preprocess_ecg(signal, fs=250):
    """
    Complete ECG preprocessing pipeline:
    1. Bandpass filter
    2. R-peak detection
    3. RR interval calculation

    Returns:
        dict with filtered_signal, r_peaks, rr_intervals_ms
    """
    filtered = bandpass_filter(signal, fs=fs)
    r_peaks = detect_r_peaks(filtered, fs=fs)
    rr_intervals_ms = compute_rr_intervals(r_peaks, fs=fs, unit="ms")

    return {
        "filtered_signal": filtered,
        "r_peaks": r_peaks,
        "rr_intervals_ms": rr_intervals_ms
    }
