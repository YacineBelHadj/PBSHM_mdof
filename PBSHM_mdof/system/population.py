import os
import pickle
import numpy as np 
import configparser

from .baseline_config import load_config_file, extract_variables
class Population:
    """ class that handels the population of systems. """    

    def __init__(self, systems:dict=dict(),name:str='POPULATION_1'):
        self.systems_params = systems
        self.name:str = name
        self.systems_matrices =dict()
        self.state:str = 'healthy'
        self.anomaly_level = 0

    def compute_systems_matrices(self):
        for key,values in self.systems_params.items():
            self.systems_matrices[key] = build_system_matrices(values)
    
    def heterogenise(self,std:float=0.01,inplace:bool=False):
        """ Heterogenises the population of systems by adding a random noise to
        the stiffness values
        Parameters
        ----------
        std : float, optional
            Standard deviation of the normal distribution used to generate the
            random noise. The default is 0.1.
        """
        if inplace==False:
            population_c = self.copy()
        else :
            population_c = self
        N=len(self.systems_params)
        k_var = np.random.normal(0,std,N)
        for i,(key,values) in enumerate(population_c.items()):
            self.systems_params[key]['stiffness'] = values['stiffness'] *(1- k_var[i])
        self.compute_systems_matrices()


    def generate_population(self):
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
            system_variables = {}
            system_variables['mass'] = np.random.normal(m_mean, m_std)
            system_variables['stiffness'] = np.random.normal(k_mean, k_std)
            system_variables['stiffness'][0]=1e-6 + 0.001*np.random.normal(0,1e-6)
            system_variables['damping'] = np.random.normal(c_mean, c_std)
            
            self.systems_params[f'system_{i}']=system_variables
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
            


def build_system_matrices(system_params: dict):
    m = system_params['mass']
    k = system_params['stiffness']
    c = system_params['damping']

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
    result = {'M':M, 'K':K, 'C':C}
    return result

if __name__ == '__main__':
    Population().generate_system_variables()