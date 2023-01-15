import h5py
import arrow
import numpy as np
from scipy.signal import welch
from PBSHM_mdof.data.data_handler import HDF5Handler

# Define the function to compute the PSDs
def compute_PSD(time_data, signal_data):
    f, psd = welch(signal_data, 1/time_data[1])
    return f, psd
psds=[]
# Open the original HDF5 file
with h5py.File('..data/raw/dataset.hdf5', 'r') as input_file:
    datahandler = HDF5Handler(input_file)
    for sys_name, state, anomaly_level, output_data in datahandler.get_data_iterator(population='*', system='*', simulation='*'):
        # process the data here
        f,psd = compute_PSD(output_data,dt=dt)
        psds.append(psd)
