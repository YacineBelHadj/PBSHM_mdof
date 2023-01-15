import numpy as np 
from scipy import signal 

def compute_PSD(signal_data:np.ndarray,fs:int):
    """
    Computes the power spectral density of a signal.
    """
    f, Pxx = signal.welch(signal_data, fs=fs, window='hann', nperseg=1024, noverlap=256)
    return f, Pxx