
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
from scipy import signal
from scipy.signal import lti
import h5py
from pathlib import Path
import nptdms
import json 
from mdof_system import mdof_system
from population import Population, build_system_matrices

def add_anomaly(system_params: dict, location: int, anomaly_type: str, anomaly_size: float = 0.1):
    """
    Adds an anomaly to the system specified in a specific location in the system_params dictionary.
    Available anomaly types: 'stiffness', 'mass', 'damping'
    """
    available_anomaly_types = {'mass':0, 'stiffness':1, 'damping':2}
    if anomaly_type not in available_anomaly_types.keys():
        raise ValueError(f"Invalid anomaly type. Choose from {available_anomaly_types.keys()}.")
    for key,values in system_params.items():
        system_params[key][available_anomaly_types[anomaly_type], location] *= (1-anomaly_size)
    return system_params

def compute_PSD(signal_data:np.ndarray,fs:int):
    """
    Computes the power spectral density of a signal.
    """
    f, Pxx = signal.welch(signal_data, fs=fs, window='hann', nperseg=1024, noverlap=256)
    return f, Pxx

def main(new_data:bool=True):
    population = Population()
    if new_data:
        population.generate_system_variables()
        population.save_systems('data/systems/systems_healthy.json')
    else: 
        population.load_systems('data//systems/systems_healthy.json')

    dt= 0.0025
    t = np.arange(0, 30, dt)
    fig,ax=plt.subplots(nrows=2,ncols=1,figsize=(15,5))

    for sys_name,sys_param in population.systems.items():
        M, K, C = build_system_matrices(sys_param)
        sys = mdof_system(M=M, K=K, C=C, sys_name=sys_name)
        t_out,y,x=sys.simulate_white_noise(t=t)
        cut = int(4/dt)
        ax[0].plot(t_out, y[:,3*7])
        ax[0].set_ylabel('Acceleration (m)')
        ax[0].set_xlabel('Time (s)')
        ax[0].grid(which='both', linestyle=':')
        ax[0].axvline(x=cut*dt, color='k', linestyle='--',label='establishement time')
        ax[1].semilogy(*compute_PSD(y[cut:,3*7],1/dt))
        ax[1].set_ylabel('PSD (m^2/Hz)')
        ax[1].set_xlabel('Frequency (Hz)')
        ax[1].grid(which='both', linestyle=':')
        plt.show()
        plt.close()      
        break



  

    



if __name__=='__main__':
    main()

# def garbage():
#     w = np.linspace(0, 120, 1000)
#     H_3f = transfer_function(M=M, K=K, C=C, omega=w,i=0,j=7)
#     fig, axes = plt.subplots(2,1)

#     axes[0].semilogy(w,np.abs(H_3f))
#     axes[0].grid(which='both', linestyle=':')
#     axes[0].set_ylabel('Magnitude H18')
#     axes[1].plot(w,np.angle(H_3f,deg=True))
#     axes[1].set_ylabel('Angle (°)')
#     axes[1].set_xlabel('Frequency (Hz)')

#     axes[1].grid(which='both', linestyle=':')
#     plt.show()
#     plt.close()

# def main():
#     dt = 0.015
#     sys=generate_system_variables() 
#     sys=system['system_1']
#     M,K,C=build_system_matrices(system)
#     #fig,ax=plt.subplots(nrows=2,ncols=1,figsize=(10,5))
#     t = np.arange(0, 60, dt)
#     u = np.zeros((len(t), 8))
#     u[:,7]=10*np.random.normal(0,10,(len(t),))
#     systems_reponse={}
#     for name,system in sys.items():
#         M,K,C=build_system_matrices(system)
#         F,G=get_state_matrices(M,K,C)
#         C_o,D= get_observation_matrices(M,K,C)
#         t,y_0=simulate_response_noise(F,G,C_o,D,t=t,u=u)
#         systems_reponse[name]=y_0
        
#         #plot_response(t,y_0,ax=ax[0],label=name)
#         #plot_PSD(y_0,1/dt,ax=ax[1],label=name,alpha=0.5)
#     p = Path.cwd() / 'data' /'raw' / f'systems_{i}_healthy.json'
#     save_systems_to_json(systems_reponse,time=t,file_name=p)
#fig.legend()
#plt.show() 