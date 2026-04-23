import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, lowcut=0.5, highcut=50, fs=250, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal
