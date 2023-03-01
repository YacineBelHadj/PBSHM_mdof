#%%
from pathlib import Path
import h5py
import numpy as np
from scipy.signal import welch
from PBSHM_mdof.data.data_handler import HDF5Handler
from tqdm import tqdm
from config import settings
# Define the function to compute the PSDs
def compute_PSD(signal_data,dt):
    f, psd = welch(signal_data, 1/dt, nperseg=1024, scaling='spectrum')
    return f, psd
#%%
psds=[]
labels=[]
anomaly = []
state=[]
latent = []
resonance_freq = []
# Open the original HDF5 file
path = Path.cwd()/settings.default['path']['generated_dataset']
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
        latent.append(d['latent_value'])
        resonance_freq.append(d['resonance_frequency'])

np.save('data/processed3/freq.npy',np.stack(f))
np.save('data/processed3/psds.npy',np.stack(psds))
np.save('data/processed3/labels.npy',np.stack(labels))
np.save('data/processed3/anomaly.npy',np.stack(anomaly))
np.save('data/processed3/states.npy',np.stack(state))
np.save('data/processed3/latent.npy',np.stack(latent))
np.save('data/processed3/resonance_freq.npy',np.stack(resonance_freq))


# %%
