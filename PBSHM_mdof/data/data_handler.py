from __future__ import annotations
from abc import ABC, abstractmethod
import h5py
import numpy as np
from typing import Dict, List, Tuple
from PBSHM_mdof.system.population import Population
from PBSHM_mdof.data.utils import resonance_frequency_computation
import h5py
from collections import defaultdict
import logging
from config import settings
from pathlib import Path
abspath = settings.default['path']['abspath']
log_path = Path(abspath)/Path('logs/loading_data.log')
class HDF5DataBuilder:
    """This class is used to build the HDF5 file containing the data.
    The data is saved in the following format:
    - POPULATION_1
        - population_params 
            - system_1
                - mass
                - stiffness
                - damping
            - system_2 

        - default_simulation
            - TDD
                - system_1  
                    - acc_1
                    - acc_2 

                - system_2
    
            setting of simulation
    """

    def __init__(self, file: h5py.File):
        self.file = file 
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        log_path = Path(settings.default['path']['abspath']) / Path('logs/loading_data.log')
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def set_save_population(self, population):
        population_group = self.file.create_group(population.name)
        population_params = population_group.create_group('population_params')
        for sys_name, sys_params in population.systems_params.items():
            sys_grp = population_params.create_group(sys_name)
            sys_grp.create_dataset('mass', data=sys_params['mass'])
            sys_grp.create_dataset('stiffness', data=sys_params['stiffness'])
            sys_grp.create_dataset('damping', data=sys_params['damping'])
        return population_group

    def get_population(self, population_group):
        data = {k:v for k,v in population_group.items()}
        return data

    def set_save_simulation_params(self, population_group, simulation_name, dt, t_end, std_latent):
        simulation_group = population_group.create_group(simulation_name)
        simulation_group.attrs['dt'] = dt
        simulation_group.attrs['t_end'] = t_end
        simulation_group.attrs['std_latent'] = std_latent
        self.logger.info(f'{population_group.name} experiment params saved in {simulation_name}')
        return simulation_group

    def get_experiment_params(self, population_group):
        dt = population_group.attrs['dt']
        t_end = population_group.attrs['t_end']
        std_latent = population_group.attrs['std_latent']
        self.logger.info(f'{population_group.name} experiment params loaded')
        return dt, t_end, std_latent

    def set_save_experiment_params(self, simulation_group, experiment_name,
                                   latent_value, anomaly_level, state, loc, amplitude):
        experiment_group = simulation_group.create_group(experiment_name)
        experiment_group.attrs['latent_value'] = str(latent_value)
        experiment_group.attrs['anomaly_level'] = str(anomaly_level)
        experiment_group.attrs['state'] = str(state)
        experiment_group.attrs['input_location'] = str(loc)
        experiment_group.attrs['amplitude'] = str(amplitude)
        self.logger.info(f'{experiment_name} params saved in {simulation_group.name}')
        return experiment_group

    def get_simulation_params(self, simulation_group):
        latent_value = simulation_group.attrs['latent_value']
        anomaly_level = simulation_group.attrs['anomaly_level']
        state = simulation_group.attrs['state']
        loc = simulation_group.attrs['input_location']
        amplitude = simulation_group.attrs['amplitude']
        self.logger.info(f'{simulation_group.name} params loaded')
        return latent_value, anomaly_level, state, loc, amplitude

    def save_experiment_population_params(self, simulation_group, population: Population):
        resonance_frequency_group = simulation_group.create_group('resonance_frequency')
        resonance_frequency_data = resonance_frequency_computation(population)
        for sys_name, freq in resonance_frequency_data.items():
            resonance_frequency_group.create_dataset(sys_name, data=freq)

        stiffness_group = simulation_group.create_group('stiffness')
        for sys_name, sys_params in population.systems_params.items():
            stiffness_group.create_dataset(sys_name, data=sys_params['stiffness'])
        self.logger.info(f'Simulated population params saved in {simulation_group.name}')

    def get_experiment_population_params(self, simulation_group):
        try:
            resonance_frequency_group = simulation_group['resonance_frequency']
            resonance_frequency_data = {sys_name: sys_data[:] for sys_name, sys_data in resonance_frequency_group.items()}
        except KeyError:
            self.logger.warning("The 'resonance_frequency' key is not present in the simulation group.")
            raise KeyError("The 'resonance_frequency' key is not present in the simulation group.")
            
        stiffness_group = simulation_group['stiffness']
        stiffness_data = {sys_name: sys_data[:] for sys_name, sys_data in stiffness_group.items()}
        self.logger.info(f'simulation {simulation_group.name} simulated params loaded')
        return resonance_frequency_data, stiffness_data


    def save_experiment_population_params(self, simulation_group, population: Population):
        resonance_frequency_group = simulation_group.create_group('resonance_frequency')
        resonance_frequency_data = resonance_frequency_computation(population)
        for sys_name, freq in resonance_frequency_data.items():
            resonance_frequency_group.create_dataset(sys_name, data=freq)

        stiffness_group = simulation_group.create_group('stiffness')
        for sys_name, sys_params in population.systems_params.items():
            stiffness_group.create_dataset(sys_name, data=sys_params['stiffness'])
        self.logger.info(f'simulation {simulation_group.name} simulated population params saved')

    def save_experiment_time_domain_data(self, simulation_group, data):
        if not isinstance(data, dict):
            raise TypeError("Data must be a dictionary.")
        
        simulation_data = simulation_group.create_group('TDD')
        for key, value in data.items():
            if not isinstance(value, dict):
                raise TypeError("Values in data dictionary must be dictionaries.")
            
            if 'output' not in value:
                raise ValueError("Missing 'output' key in value dictionary.")
            
            output = value['output']
            if not isinstance(output, np.ndarray):
                raise TypeError("Output must be a NumPy array.")
            
            simulation_data.create_dataset(key, shape=output.shape, dtype=output.dtype, data=output)
        
        self.logger.info(f'simulation {simulation_group.name} time domain data saved')
        return simulation_group


    def get_experiment_time_domain_data(self, simulation_group):
            if 'TDD' not in simulation_group:
                self.logger.warning("The 'TDD' key is not present in the simulation group.")
                return {}
            simulation_data = simulation_group['TDD']
            data = {}
            for key, value in simulation_data.items():
                data[key] = value[:]
            self.logger.info(f'simulation {simulation_group.name} time domain data loaded')
            return data


    def save_experiment_frequency_domain_data(self, simulation_group, data):
        simulation_data = simulation_group.create_group('FDD')
        for key, value in data.items():
            simulation_data.create_dataset(key, shape=value.shape, dtype=value.dtype, data=value)
        self.logger.info(f'simulation {simulation_group.name} frequency domain data saved')

   


    def get_experiment_frequency_domain_data(self, simulation_group):
        if 'FDD' not in simulation_group:
            self.logger.warning("The 'FDD' key is not present in the simulation group.")
            return {}
        simulation_data = simulation_group['FDD']
        data = {}
        for key, value in simulation_data.items():
            data[key] = value[:]
        self.logger.info(f'simulation {simulation_group.name} frequency domain data loaded')
        return data


    def get_simulation_dt(self, population_name, simulation_name):
        population_group = self.file[population_name]
        simulation_group = population_group[simulation_name]
        dt = simulation_group.attrs['dt']
        self.logger.info(f'population {population_name} simulation {simulation_name} dt loaded')
        return dt
    
    def get_simulation_latent_std(self, population_name, simulation_name):
        population_group = self.file[population_name]
        simulation_group = population_group[simulation_name]
        latent_std = simulation_group.attrs['latent_std']
        self.logger.info(f'population {population_name} simulation {simulation_name} latent_std loaded')
        return latent_std

    
    def get_data_iterator(self):
        for pop_name in self.file.keys():
            pop_group = self.file[pop_name]
            for sim_name in pop_group.keys():
                sim_group = pop_group[sim_name]
                sim_params = dict(sim_group.attrs.items())
                for exp_name in sim_group.keys():
                    try:
                        exp_group = sim_group[exp_name]
                        exp_params = dict(exp_group.attrs.items())
                        
                        resonance_freq_data, stiffness_data = self.get_experiment_population_params(exp_group)
                        time_domain_data = self.get_experiment_time_domain_data(exp_group)

                        result ={'population_name':pop_name,'simulation_name':sim_name,'experiment_name':exp_name,
                                'simulation_params':sim_params,'experiment_params':exp_params,
                                'resonance_frequency_data':resonance_freq_data,'stiffness_data':stiffness_data,
                                'TDD':time_domain_data}

                        yield result
                    except :
                        self.logger.warning(f"Error while loading data for population {pop_name}, simulation {sim_name}, experiment {exp_name}")
                        break

    def close(self):
        self.file.close()
        
    def check_data_existence(self, population_name=None, system_name=None, simulation_name=None):
        if not population_name:
            # check if any population exists
            if len(self.file.keys()) == 0:
                return False
            return True
        elif population_name not in self.file:
            return False

        population_group = self.file[population_name]
        if not system_name:
            # check if any system exists
            if len(population_group.keys()) == 1 and 'population_params' in population_group:
                return False
            return True
        elif system_name not in population_group:
            return False

        system_group = population_group[system_name]
        if not simulation_name:
            # check if any simulation exists
            if len(system_group.keys()) == 0:
                return False
            return True
        elif simulation_name not in system_group:
            return False

        return True



if __name__ =='__main__':
    path_raw= Path(settings.default['path']['abspath']) / Path(settings.default['path']['generated_dataset'])
    with h5py.File(path_raw, 'r') as f:
        dh = HDF5DataBuilder(f)
        iter_=dh.iter_data()
        for i in iter_:
            print(i['simulation_data']['resonance_frequency']['system_1'])
            break





