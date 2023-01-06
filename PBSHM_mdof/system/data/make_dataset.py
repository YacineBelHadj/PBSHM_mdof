
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
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)

def generate_system_variables(N:int=20, 
    m_mean:np.ndarray = 5*np.array([0.5318, 0.4040, 0.4101, 0.4123, 0.3960, 0.3809, 0.4086, 0.3798]),
    k_mean:np.ndarray = 3e2*np.array([56.7, 56.70, 56.70, 56.70, 56.70, 56.70, 56.70, 56.70]),
    c_mean:np.ndarray = np.array([8.746, 8.791, 8.801, 8.851, 8.714, 8.737, 8.549, 8.752]),
    m_std:float = 0.05, 
    k_std:float=10, 
    c_std:float=0.08):

    # Generate N sets of system parameters with random noise
    system_params = dict()
    for i in range(N):
        m_i = np.random.normal(m_mean, m_std)
        k_i = np.random.normal(k_mean, k_std)
        c_i = np.random.normal(c_mean, c_std)
        system_params[f'system_{i}']=np.stack((m_i, k_i, c_i))
    return system_params

def build_system_matrices(system_params: np.ndarray):
    m = system_params[0]
    k = system_params[1]
    c = system_params[2]

    # Number of masses
    n = m.shape[0]

    # Initialize mass matrix
    M = np.diag(m)
    # Initialize stiffness matrix
    K = np.zeros((n, n))
    for i in range(n-1):
        K[i, i] = k[i] + k[i+1]
        K[i, i+1] = K[i+1, i] = -k[i+1]

    K[-1, -1] = k[-1]
    
    # Initialize damping matrix
    #C = np.zeros((n, n))
    #for i in range(n-1):
    #    C[i, i] = c[i] + c[i+1]
    #    C[i, i+1] = C[i+1, i] = -c[i+1]

    #C[-1, -1] = c[-1]
    C = K *1/1e5
    return M, K, C

def check_rank_matrix(system: Tuple[np.ndarray]):
    """
    Checks if the matrix M is full rank.
    """
    for matrix in system:
        assert(np.linalg.matrix_rank(matrix) == matrix.shape[0])

def compute_modes(M: np.ndarray, K: np.ndarray, C: np.ndarray, n_modes: int = 3):
    """
    Computes the first n_modes of a system with mass matrix M, stiffness matrix K and damping matrix C.
    """
    # Solve eigenvalue problem
    omega_temp, phi = sp.linalg.eig(K,M)

    idx = np.argsort(omega_temp)
    # Sort eigenvalues and eigenvectors
    omega = omega_temp[idx]
    phi = phi[:, idx]

    omega = np.sqrt(omega)

    # Compute mode shapes
    return omega, phi



def check_orthogonality(M:np.ndarray,phi: np.ndarray):
    """
    Checks if the mode shapes are orthogonal.
    """
    mu = phi.T@M@phi
    np.allclose(mu-np.diag(np.diagonal(mu),0), np.zeros((M.shape[0], M.shape[0])))

def project_modal(M:np.ndarray, K:np.ndarray, C:np.ndarray, phi:np.ndarray):
    """
    Projects the system matrices onto the modal basis.
    """
    M_modal = phi.T@M@phi
    K_modal = phi.T@K@phi
    C_modal = phi.T@C@phi
    return M_modal, K_modal, C_modal

def is_diagonal(matrix: np.ndarray) -> bool:
    """
    Returns True if the matrix is diagonal, False otherwise.
    """
    # Check if matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False
    
    # Check if all off-diagonal elements are 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i != j and np.abs(matrix[i, j]) > 1e-6:
                return False
    return True

def sdof_impulse_response(m:float,c:float,k:float,t:np.ndarray):

    xi= c/(2*np.sqrt(m*k))
    w_n = np.sqrt(k/m)
    w_n = np.sqrt(k/m)
    w_d = w_n*np.sqrt(1-xi**2)  
    h=1/(m*w_d)*np.exp(-xi*w_n*t)*np.sin(w_d*t)
    return h

def mdof_impulse_response(M:np.ndarray, K:np.ndarray, C:np.ndarray, phi:np.ndarray, t:np.ndarray):
    """
    Computes the impulse response of a system with mass matrix M, stiffness matrix K, damping matrix C and mode shapes phi.
    """
    M_modal, K_modal, C_modal = project_modal(M, K, C, phi)
    # Check if the matrices are diagonal


    # Initialize impulse response
    h = np.zeros((phi.shape[0], t.shape[0]))

    # Compute impulse response
    for i in range(phi.shape[0]):
        h[i, :] = sdof_impulse_response(M_modal[i, i], C_modal[i, i], K_modal[i, i], t)
    return h

def transfer_function(M:np.ndarray, K:np.ndarray, C:np.ndarray, omega:np.ndarray,i:int,j:int):
    s= 1j*omega
    n_dof= M.shape[0]
    M_s2 = np.einsum('ij,k->ijk', M, s**2)
    C_s = np.einsum('ij,k->ijk', C, s**1)
    K_s = np.einsum('ij,k->ijk', K, s**0)

    # Initialize the array to store the transfer function
    H_ij = []

    # Compute the transfer function for each frequency
    for s_i in range(len(s)):
        H_ij.append(np.linalg.inv((M_s2+C_s+K_s)[:,:,s_i])[i,j])

    return H_ij


def get_state_matrices(M: np.ndarray, K: np.ndarray, C: np.ndarray):
    n_dof = M.shape[0]
    F = np.block([[np.zeros((n_dof, n_dof)), np.eye(n_dof)],
                  [-np.linalg.inv(M) @ K, -np.linalg.inv(M) @ C]])
    G = np.block([[np.zeros((n_dof, n_dof))], [np.linalg.inv(M)]])
    return F, G

from scipy.linalg import expm

def discretize(F: np.ndarray, G: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    A = expm(F * dt)
    B = np.linalg.inv(F).dot((A - np.eye(F.shape[0]))).dot(G)
    return A, B


def get_observation_matrices(M: np.ndarray, K: np.ndarray, C: np.ndarray):
    n_dof = M.shape[0]
    C = np.block([[np.eye(n_dof), np.zeros((n_dof, n_dof))],
                  [np.zeros((n_dof, n_dof)), np.eye(n_dof)],
                  [-K@np.linalg.inv(M), -C@np.linalg.inv(M)]])
    D = np.block([[np.zeros((n_dof, n_dof))], [np.zeros((n_dof, n_dof))], [np.eye(n_dof)@np.linalg.inv(M)]])
    return C, D




import numpy as np

def simulate_response(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, 
        u: np.ndarray, x0: np.ndarray, time:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_steps = len(time)
    n_outputs = C.shape[0]
    n_states = A.shape[0]

    x = np.zeros((n_steps, n_states))
    y = np.zeros((n_steps, n_outputs))

    x[0, :] = x0
    for i in range(1, n_steps):
        x[i, :] = A.dot(x[i-1, :]) + B.dot(u[i-1, :])
        y[i, :] = C.dot(x[i, :]) + D.dot(u[i-1, :])
    return y

def plot_PSD(y: np.ndarray, fs: float, nfft: int = 1024, overlap: int = 0, window: str = 'hann', 
        color: str = 'blue', label: str = None, ax: plt.Axes = None, **kwargs):
    """
    Plots the power spectral density of a signal.
    """
    if ax is None:
        fig, ax = plt.subplots()
    f, Pxx = signal.welch(y, fs=fs, nfft=nfft, window=window, noverlap=overlap, **kwargs)
    ax.semilogy(f, Pxx, color=color, label=label)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('PSD [V**2/Hz]')
    ax.grid(True)
    return ax

def simulate_response_noise(F:np.ndarray,G:np.ndarray,C:np.ndarray,D:np.ndarray,u:np.ndarray,t:np.ndarray):
    x0 = np.zeros(F.shape[0])

    sys = lti(F, G, C, D)
    t, y, x = signal.lsim(sys, U=u,X0=x0, T=t)
    y_0 = y[:,0]
    # add noise
    noise_rms = np.sqrt(np.mean(np.square(y_0)))/160
    y_0 += np.random.normal(0,noise_rms,(len(t),))  
    return t, y_0

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

def plot_response(t: np.ndarray, y: np.ndarray, ax: plt.Axes = None, **kwargs):
    """
    Plots the response of a system.
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(t, y, **kwargs)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Displacement [m]')
    ax.grid(True)
    return ax
def plot_PSD(y: np.ndarray, fs: float,nperseg=1500, overlap: int = 5*256, window: str = 'hann',
 label: str = None, ax: plt.Axes = None,alpha:float=1, **kwargs):
    """
    Plots the power spectral density of a signal.
    """
    if ax is None:
        fig, ax = plt.subplots()
    f, Pxx = signal.welch(y, fs=fs, window=window,nperseg=nperseg, noverlap=overlap, **kwargs)
    ax.semilogy(f, Pxx, label=label,alpha=alpha)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('PSD [V**2/Hz]')
    ax.grid(True)
    return ax


def save_systems_to_hdf5(systems: dict, anomaly: float, time: np.ndarray, group_name: str, file_name: str):
    # Open the file in read/write mode
    with h5py.File(file_name, 'w+') as h5file:
        # Create a group for the systems
        systems_group = h5file.create_group(group_name)
        # Create a dataset for the time data
        systems_group.create_dataset('time', data=time)
        # Iterate over the systems
        for system_id, output in systems.items():
            # Create a group for the system
            system_group = systems_group.create_group(system_id)
            # Create a dataset for the output data
            system_group.create_dataset('output', data=output)
            # Set the anomaly level for the system
            system_group.attrs['anomaly_level'] = anomaly

def save_systems_to_tdms(systems: dict, time: np.ndarray, file_name: str):
    # Create a TdmsWriter object
    writer = nptdms.TdmsWriter(file_name)
    
    # Create a group for the systems
    systems_group = writer.group('systems')
    
    # Create a channel for the time data
    time_channel = systems_group.channel('time')
    
    # Write the time data to the channel
    time_channel.write(time)
    
    # Iterate over the systems
    for system_id, output in systems.items():
        # Create a group for the system
        system_group = systems_group.group(system_id)
        # Create a channel for the output data
        output_channel = system_group.channel('output')
        # Write the output data to the channel
        output_channel.write(output)

def main():
    for i in range(1800):
        dt = 0.015
        sys=generate_system_variables() 
        #fig,ax=plt.subplots(nrows=2,ncols=1,figsize=(10,5))
        t = np.arange(0, 60, dt)
        u = np.zeros((len(t), 8))
        u[:,7]=10*np.random.normal(0,10,(len(t),))
        systems_reponse={}
        for name,system in sys.items():
            M,K,C=build_system_matrices(system)
            F,G=get_state_matrices(M,K,C)
            C_o,D= get_observation_matrices(M,K,C)
            t,y_0=simulate_response_noise(F,G,C_o,D,t=t,u=u)
            systems_reponse[name]=y_0
            
            #plot_response(t,y_0,ax=ax[0],label=name)
            #plot_PSD(y_0,1/dt,ax=ax[1],label=name,alpha=0.5)
        p = Path.cwd() / 'data' /'raw' / f'systems_{i}_healthy.tdms'
        save_systems_to_tdms(systems_reponse,time=t,file_name=p)
    #fig.legend()
    #plt.show()
    #plt.close()

  

    



if __name__=='__main__':
    main()

def garbage():
    w = np.linspace(0, 120, 1000)
    H_3f = transfer_function(M=M, K=K, C=C, omega=w,i=0,j=7)
    fig, axes = plt.subplots(2,1)

    axes[0].semilogy(w,np.abs(H_3f))
    axes[0].grid(which='both', linestyle=':')
    axes[0].set_ylabel('Magnitude H18')
    axes[1].plot(w,np.angle(H_3f,deg=True))
    axes[1].set_ylabel('Angle (°)')
    axes[1].set_xlabel('Frequency (Hz)')

    axes[1].grid(which='both', linestyle=':')
    plt.show()
    plt.close()
