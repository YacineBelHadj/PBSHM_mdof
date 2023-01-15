
from scipy.signal import welch, decimate, detrend
import numpy as np
import sys
EPS = sys.float_info.epsilon

def compute_PSD(signals:np.ndarray,fs:int=250,q:int=1,tperseg:int=30,toverlap:int=15):
    """ Compute the power spectral density of the signal with Welch's method.
        with a decimate factor of q. if q=1, no decimation is performed.
    Args:
        signals (np.ndarray): _description_
        fs (int, optional): _description_. Defaults to 250.
        q (int, optional): _description_. Defaults to 2.
        tperseg (int, optional): _description_. Defaults to 250.
        toverlap (int, optional): _description_. Defaults to 250.


    Returns:
        _type_: _description_
    """
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    if q > 1:
        signals = decimate(signals, q, axis=1)
    signals = detrend(signals,type='constant')
    fs = int(fs / q)
    signals = signals - np.mean(signals, axis=1, keepdims=True)
    f,Sxxs= welch(signals,fs=fs,nperseg=fs*tperseg,noverlap=fs*toverlap)
    return f,Sxxs