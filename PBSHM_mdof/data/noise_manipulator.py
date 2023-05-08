
import numpy as np

def noise_std(SNR_dB:int,RMS_signal:float=15):
    SNR_lin = 10**(SNR_dB/20)
    RMS_noise = RMS_signal/SNR_lin
    std_noise = RMS_noise
    return std_noise

def generate_noise(SNR_dB:int,signal_length:int,RMS_signal:float = 15):
    std_noise = noise_std(SNR_dB,RMS_signal=RMS_signal)
    noise = np.random.normal(0,std_noise,signal_length)
    return noise
