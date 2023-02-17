#%%
from pathlib import Path
import h5py
import numpy as np
from scipy.signal import welch
from PBSHM_mdof.data.data_handler import HDF5Handler
from tqdm import tqdm

# Define the function to compute the PSDs
def compute_PSD(signal_data,dt):
    f, psd = welch(signal_data, 1/dt, nperseg=2048, scaling='spectrum')
    return f, psd
#%%
psds=[]
labels=[]
anomaly = []
state=[]
latent = []
# Open the original HDF5 file
path = Path.cwd()/'data'/'raw'/'datatest.hdf5'
print(path)
assert path.exists()
with h5py.File(path, 'r') as input_file:
    dh = HDF5Handler(input_file)
    # Loop over the simulations
    iterator=dh.get_data_iterator()
    for d in tqdm(iterator,total=2600*20):
        # Compute the PSDs
        acc_7=d['output'][:,2*8+1]
        f, psd = compute_PSD(acc_7, d['dt'])
        psds.append(psd)
        labels.append(d['system_name'])
        anomaly.append(d['anomaly_severity'])
        state.append(d['state'])
        latent.append(d['temperature'])

np.save('data/processed2/freq.npy',np.stack(f))
np.save('data/processed2/psds.npy',np.stack(psds))
np.save('data/processed2/labels.npy',np.stack(labels))
np.save('data/processed2/anomaly.npy',np.stack(anomaly))
np.save('data/processed2/states.npy',np.stack(state))
np.save('data/processed2/latent.npy',np.stack(latent))
# %%
