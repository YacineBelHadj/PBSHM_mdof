from dataclasses import dataclass
from functools import lru_cache
from scipy.linalg import expm
from scipy.signal import lti , lsim
import numpy as np

import checks
import pandas as pd 
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)
@dataclass
class mdof_system:
    M: np.ndarray 
    K: np.ndarray
    C: np.ndarray
    sys_name: str = 'mdof_system'

    def __post_init__(self):
        self.n_dof = self.M.shape[0]
        self.check_system()

  
    @property
    def A(self):
        return np.block([[np.zeros((self.n_dof, self.n_dof)), np.eye(self.n_dof)],
                    [-np.linalg.inv(self.M) @ self.K, -np.linalg.inv(self.M) @ self.C]])
    
    @property
    def B(self):
        return np.block([[np.zeros((self.n_dof, self.n_dof))],
                     [np.linalg.inv(self.M)]])
    
    @property
    def C_o(self):
        return np.block([[np.eye(self.n_dof), np.zeros((self.n_dof, self.n_dof))],
                    [np.zeros((self.n_dof, self.n_dof)), np.eye(self.n_dof)],
                    [-np.linalg.inv(self.M) @ self.K, -np.linalg.inv(self.M) @ self.C]])
    
    @property
    def D(self):
        return np.block([[np.zeros((self.n_dof, self.n_dof))],
         [np.zeros((self.n_dof, self.n_dof))], 
         [np.eye(self.n_dof)@np.linalg.inv(self.M)]])

    @property
    def A_discrete(self):
        return expm(self.A * self.dt)

    @property
    def B_discrete(self):
        return np.linalg.solve(self.A_discrete - self.A, self.B)

    def __repr__(self):

        name = self.sys_name
        return ('name = {} \n\n'
                'Mass Matrix: \n'
                '{} \n\n'
                'Stiffness Matrix: \n'
                '{} \n\n'
                'Damping Matrix: \n'
                '{}'.format(name,pd.DataFrame(self.M), pd.DataFrame(self.K), pd.DataFrame(self.C)))
    
    def check_system(self):
        assert checks.is_diagonal(self.M), "C is not diagonal"
        assert checks.check_rank_matrix((self.M, self.K, self.C)), "M or K is not full rank"

    def transfer_function(self,omega:np.ndarray,i:int,j:int):
        s= 1j*omega
        M_s2 = np.einsum('ij,k->ijk', self.M, s**2)
        C_s = np.einsum('ij,k->ijk', self.C, s**1)
        K_s = np.einsum('ij,k->ijk', self.K, s**0)

        # Initialize the array to store the transfer function
        H_ij = []

        # Compute the transfer function for each frequency
        for s_i in range(len(s)):
            H_ij.append(np.linalg.inv((M_s2+C_s+K_s)[:,:,s_i])[i,j])

        return H_ij

    def simulate_homemade(self, t, u, x0=None):
        nsamples = len(t)
        noutputs = self.C_o.shape[0]
        x = np.zeros((nsamples, self.n_dof))
        y = np.zeros((nsamples, noutputs))

        x[0,:] = x0 if x0 is not None else np.zeros(self.n_dof)
        for i in range(1,nsamples):
            x[i,:] = self.A_discrete.dot(x[i-1,:]) + self.B_discrete.dot(u[i,:])
            y[i,:] = self.C_o.dot(x[i,:]) + self.D.dot(u[i,:])
        return y

    def simulate_lsim(self, u:np.ndarray, t:np.ndarray, x0=None):
        sys = lti(self.A, self.B, self.C_o, self.D)
        return lsim(sys, u, t)

    def simulate_white_noise(self,t,location:int = 7):
        nsamples = len(t)
        amp = np.random.uniform(0.2,0.4)
        scale = np.random.uniform(1,2)
        u = np.zeros((nsamples,self.n_dof))
        u[:,location] = amp*np.random.normal(0,scale,(nsamples,))
        return self.simulate_lsim(t=t,u=u)
    


# if False:     
#     def garbage():
#         pass


#     def plot_response(t: np.ndarray, y: np.ndarray, ax: plt.Axes = None, **kwargs):
#         """
#         Plots the response of a system.
#         """
#         if ax is None:
#             fig, ax = plt.subplots()
#         ax.plot(t, y, **kwargs)
#         ax.set_xlabel('Time [s]')
#         ax.set_ylabel('Displacement [m]')
#         ax.grid(True)
#         return ax
#     def plot_PSD(y: np.ndarray, fs: float,nperseg=1500, overlap: int = 5*256, window: str = 'hann',
#     label: str = None, ax: plt.Axes = None,alpha:float=1, **kwargs):
#         """
#         Plots the power spectral density of a signal.
#         """
#         window = signal.window.get_window(window, nperseg)
        
#         if ax is None:
#             fig, ax = plt.subplots()
#         f, Pxx = signal.welch(y, fs=fs, window=window,nperseg=nperseg, noverlap=overlap, **kwargs)
#         ax.semilogy(f, Pxx, label=label,alpha=alpha)
#         ax.set_xlabel('Frequency [Hz]')
#         ax.set_ylabel('PSD [V**2/Hz]')
#         ax.grid(True)
#         return ax

#     def save_systems_to_json(systems: dict, time: np.ndarray,  file_name: str):
#         # Create a dictionary to store the data
#         for key, value in systems.items():
#             systems[key] = value.tolist()

#         systems['time'] = time.tolist()
#         # Save the data to a JSON file
#         with open(file_name, 'w') as f:
#             json.dump(systems, f)

#     def save_systems_to_h5df(systems: dict, time: np.ndarray, file_name: str):
#         # Create a dictionary to store the data
#         for key, value in systems.items():
#             systems[key] = value.tolist()

#         systems['time'] = time.tolist()
#         # Save the data to a JSON file
#         with h5py.File(file_name, 'w') as f:
#             for key, value in systems.items():
#                 f.create_dataset(key, data=value)

#     def save_systems_to_tdms(systems: dict, time: np.ndarray, file_name: str):
#         with nptdms.TdmsFile.open(file_name, mode='w+') as tdms_file:
#             systems_group = tdms_file.object().create_group('signals')
#             # Create a channel for the time data
#             time_channel = systems_group.create_channel('time', data=time)
#             # Iterate over the systems
#             for system_id, output in systems.items():
#                 # Create a group for the system
#                 system_group = systems_group.create_group(system_id)
#                 # Create a channel for the output data
#                 output_channel = system_group.create_channel('output', data=output)
        
#     def discretize(F: np.ndarray, G: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
#         A = expm(F * dt)
#         B = np.linalg.inv(F).dot((A - np.eye(F.shape[0]))).dot(G)
#         return A, B




#     def simulate_response(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, 
#             u: np.ndarray, x0: np.ndarray, time:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         n_steps = len(time)
#         n_outputs = C.shape[0]
#         n_states = A.shape[0]

#         x = np.zeros((n_steps, n_states))
#         y = np.zeros((n_steps, n_outputs))

#         x[0, :] = x0
#         for i in range(1, n_steps):
#             x[i, :] = A.dot(x[i-1, :]) + B.dot(u[i-1, :])
#             y[i, :] = C.dot(x[i, :]) + D.dot(u[i-1, :])
#         return y

#     def plot_PSD(y: np.ndarray, fs: float, nfft: int = 1024, overlap: int = 0, window: str = 'hann', 
#             color: str = 'blue', label: str = None, ax: plt.Axes = None, **kwargs):
#         """
#         Plots the power spectral density of a signal.
#         """
#         if ax is None:
#             fig, ax = plt.subplots()
#         f, Pxx = signal.welch(y, fs=fs, nfft=nfft, window=window, noverlap=overlap, **kwargs)
#         ax.semilogy(f, Pxx, color=color, label=label)
#         ax.set_xlabel('Frequency [Hz]')
#         ax.set_ylabel('PSD [V**2/Hz]')
#         ax.grid(True)
#         return ax

#     def simulate_response_noise(F:np.ndarray,G:np.ndarray,C:np.ndarray,D:np.ndarray,u:np.ndarray,t:np.ndarray):
#         x0 = np.zeros(F.shape[0])

#         sys = lti(F, G, C, D)
#         t, y, x = signal.lsim(sys, U=u,X0=x0, T=t)
#         y_0 = y[:,0]
#         # add noise
#         #noise_rms = np.sqrt(np.mean(np.square(y_0)))/160
#         #y_0 += np.random.normal(0,noise_rms,(len(t),))  
#         return t, y_0


