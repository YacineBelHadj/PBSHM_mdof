import os
import pickle
import numpy as np 

class Population:
    def __init__(self, systems:dict=dict()):
        self.systems = systems

    def generate_system_variables(self, N:int=20, 
        m_mean:np.ndarray = np.array([0.5318, 0.4040, 0.4101, 0.4123, 0.3960, 0.3809, 0.4086, 0.3798]),
        k_mean:np.ndarray = 1e3*np.array([1e-6, 56.70, 56.70, 56.70, 56.70, 56.70, 56.70, 56.70]),
        c_mean:np.ndarray = np.array([8.746, 8.791, 8.801, 8.851, 8.714, 8.737, 8.549, 8.752]),
        m_std:float = 0.05, 
        k_std:float=0.1, 
        c_std:float=0.08):

        # Generate N sets of system parameters with random noise
        for i in range(N):
            m_i = np.random.normal(m_mean, m_std)
            k_i = np.random.normal(k_mean, k_std)
            k_i[0]=1e-6
            c_i = np.random.normal(c_mean, c_std)
            self.systems[f'system_{i}']=np.stack((m_i, k_i, c_i))

    def save_systems(self, filepath):
        """
        Saves the system parameters to the specified filepath.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.systems, f)

    def load_systems(self, filepath):
        """
        Loads the system parameters from the specified filepath.
        """
        with open(filepath, 'rb') as f:
            self.systems = pickle.load(f)

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