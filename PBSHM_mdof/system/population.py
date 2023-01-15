import os
import pickle
import numpy as np 
import configparser

from .baseline_config import load_config_file, extract_variables
class Population:
    """ class that handels the population of systems. """
    AVAILABLE_ANOMALY_TYPES = {'mass':0, 'stiffness':1, 'damping':2}
    

    def __init__(self, systems:dict=dict()):
        self.systems_params = systems
        self.systems_matrices =dict()
        self.state:str = 'healthy'
        self.anomaly_level = 0
        self.name:str = 'POPULATION_1' 

    def compute_systems_matrices(self):
        for key,values in self.systems_params.items():
            self.systems_matrices[key] = build_system_matrices(values)

    def generate_system_variables(self):
        """ Generates the system parameters for the population of systems.
        The parameters are generated from a normal distribution with mean and
        standard deviation specified in the config file.
        Parameters
        ----------

        population_name : str, optional

        """

        config= load_config_file()
        N,N_dof,m_mean, m_std,k_mean,k_std,c_mean,c_std = extract_variables(config,
        population_name=self.name)
        # Generate N sets of system parameters with random noise
        for i in range(N):
            m_i = np.random.normal(m_mean, m_std)
            k_i = np.random.normal(k_mean, k_std)
            k_i[0]=1e-6 + 0.001*np.random.normal(0,1e-6)
            c_i = np.random.normal(c_mean, c_std)
            self.systems_params[f'system_{i}']=np.stack((m_i, k_i, c_i))
        self.compute_systems_matrices()

    def add_anomaly(self, location: int=5, anomaly_type: str='stiffness', anomaly_size: float = 0.1):
        """
        Adds an anomaly to the system specified in a specific location in the system_params dictionary.
        Available anomaly types: 'stiffness', 'mass', 'damping'
        """
        self.anomaly_level = anomaly_size

        self.state = 'anomalous'

        available_anomaly_types = {'mass':0, 'stiffness':1, 'damping':2}
        if anomaly_type not in available_anomaly_types.keys():
            raise ValueError(f"Invalid anomaly type. Choose from {available_anomaly_types.keys()}.")
        for key,values in self.systems_params.items():
            arr = np.copy(self.systems_params[key][available_anomaly_types[anomaly_type]])
            arr[location] *= (1-anomaly_size)
            self.systems_params[key][available_anomaly_types[anomaly_type]] = arr
        self.compute_systems_matrices()

    def load_population(self, path:str):
        """ Loads a population of systems from a pickle file.
        Parameters
        ----------
        path : str
            Path to the pickle file.
        """
        with open(path, 'rb') as f:
            self.systems_params = pickle.load(f)
        self.compute_systems_matrices()
    
    def save_population(self, path:str):
        """ Saves a population of systems to a pickle file.
        Parameters
        ----------
        path : str
            Path to the pickle file.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.systems_params, f)
            


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
    # damping matrix
    C = K *1/1e5
    return M, K, C

if __name__ == '__main__':
    Population().generate_system_variables()