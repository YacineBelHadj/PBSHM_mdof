from PBSHM_mdof.system.mdof_system import MdofSystem
from PBSHM_mdof.system.population import Population

def resonance_frequency_computation(population:Population):
    res_freq = {}
    for sys_name,sys_param in population.systems_matrices.items():
        sys = MdofSystem(**sys_param)
        res_freq[sys_name] = sys.resonance_frequency()
    return res_freq

def data_name_SNR_nperseg(SNR=None,nperseg=1024):
    if SNR == None:
        data_name = f'no_noise_nperseg_{nperseg}.parquet'
    else:
        data_name = f'SNR_{SNR}_nperseg_{nperseg}.parquet'
    return data_name