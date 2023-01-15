#%%
import h5py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal 
def compute_PSD(signal_data:np.ndarray,fs:int):
    """
    Computes the power spectral density of a signal.
    """
    f, Pxx = signal.welch(signal_data, fs=fs, window='hann', nperseg=1024, noverlap=256)
    return f, Pxx
#%%
def main(path:str):
    X =[]
    labels=[]
    state = []

    with h5py.File(path, 'r') as f:
        time_data = f['time'][:]
        dt = f['time'].attrs['dt']
        time_units = f['time'].attrs['units']
        population_data = f['population']
        population_state = population_data.attrs['state']
        for sys_name in population_data:
            system_group = population_data[sys_name]
            masses = system_group.attrs['masses']
            stiffness = system_group.attrs['stiffness']
            damping = system_group.attrs['damping']
            units = system_group.attrs['units']
            for simulation_group_name in system_group:
                simulation_group = system_group[simulation_group_name]
                output_data = simulation_group['output'][:]
                input_data = simulation_group['input'][:]
                X.append(output_data[:,8*2+1])
                labels.append(sys_name)
                state.append(population_state)
    return time_data,X, labels , state 



if __name__ == '__main__':
    path = Path('data/raw/healthy.hdf5')
    t,X1,y1,state1= main(path)
    path = Path('data/raw/anoumalous_014.hdf5')
    t,X2,y2,state2= main(path)
    X = np.concatenate((X1,X2),axis=0)
    y = np.concatenate((y1,y2),axis=0)
    state = np.concatenate((state1,state2),axis=0)
    print(y.shape)
    print(len(X))
    dt = t[1]-t[0]
    cut = int(4/dt)
    fig,ax = plt.subplots(nrows=2, ncols=1,figsize=(10,5))
    for i in range(0,200,1):
        if y[i] == 'system_0':
            ax[0].plot(t, X[i],alpha=0.5)
            ax[0].set_ylabel('Acceleration (m)')
            ax[0].set_xlabel('Time (s)')
            ax[0].grid(which='both', linestyle=':')
            ax[0].axvline(x=cut*dt, color='k', linestyle='--',label='establishement time')
            ax[1].semilogy(*compute_PSD(X[i],1/dt),alpha=0.5,label=state[i])
            ax[1].set_ylabel('PSD (m^2/Hz)')
            ax[1].set_xlabel('Frequency (Hz)')
            ax[1].grid(which='both', linestyle=':')
    
    plt.show()
    plt.close()
    

# %%
