from __future__ import annotations
from abc import ABC, abstractmethod
import h5py
import numpy as np
from typing import Dict, List, Tuple
from PBSHM_mdof.system.population import Population
import h5py

class HDF5DataBuilder:
    def __init__(self, file: h5py.File):
        self.file = file
        
    def save_population(self, population_names):
        for name in population_names:
            population_group = self.file.create_group(name)
            population_group.attrs['units'] = 'kg, N/m, Ns/m'
        
        
    def save_population_params(self, population):
        for population_name, population_params in population.system_params.items():
            system = self.file[population_name].create_group(population_name)
            system.attrs['mass'] = population_params['mass']
            system.attrs['stiffness'] = population_params['stiffness']
            system.attrs['damping'] = population_params['damping']
    
    def save_experiment_params(self, population_name, system_name, dt, t_end):
        system = self.file[population_name][system_name]
        system.attrs['dt'] = dt
        system.attrs['t_end'] = t_end
    
    def save_simulation_data(self, population_name, system_name, simulation_idx,data):
        system = self.file[population_name][system_name]
        simulation = system.create_group(simulation_idx)
        simulation.create_dataset(data, shape=data.shape, dtype=data.dtype, data=data)
            
    def set_resonance_frequency(self, population_name, resonance_frequency):
        self.file[population_name].attrs['resonance_frequency'] = resonance_frequency
    
    def set_latent_value(self, population_name, latent_value):
        self.file[population_name].attrs['latent_value'] = 
        
class H5pyHDF5SimulationBuilder:
    def __init__(self, file: h5py.File):
        self.file = file
        
    def save_simulation_data(self, population_name, system_name, simulation_name, dt, t_end, output_data, input_data, latent_value):
        population_ds = self.file[population_name]
        simulation_ds = population_ds[system_name].create_group(simulation_name)
        simulation_ds.attrs['latent_value'] = latent_value
        simulation_ds.attrs['dt'] = dt
        simulation_ds.attrs['t_end'] = t_end
        simulation_ds.create_dataset('output', shape=output_data.shape, dtype=output_data.dtype, data=output_data)
        simulation_ds.create_dataset('input', shape=input_data.shape, dtype=input_data.dtype, data=input_data)
    
    def get_data_iterator(self):
        for population_name in self.file:
            population_group = self.file[population_name]
            for system_name in population_group:
                system_group = population_group[system_name]
                for simulation_name in system_group:
                    simulation_group = system_group[simulation_name]
                    if 'output' in simulation_group:
                        res = {'output':simulation_group['output'][:],
                               'dt':simulation_group.attrs['dt'],
                               't_end':simulation_group.attrs['t_end'],
                               'population_name':population_name,
                               'system_name':system_name,
                               'latent_value':simulation_group.attrs['latent_value'],
                               'resonance_frequency':population_group.attrs['resonance_frequency']}
                        yield res
    
    def get_data(self, population:str='*', system:str='*', simulation:str='*'):
        """
        Get the output data, system name, state, and anomaly_severity for all the simulations in the HDF5 file.
        Returns:
            A list of dictionaries, where each dictionary contains the following information for a single simulation:
                {'system_name': system_name, 'state': state, 'anomaly_severity': anomaly_severity, 'output': output_data}
        """
        simulation_data = []
        for population_name in self.file:
            if population != '*' and population_name != population:
                continue
            population_group = self.file[population_name]
            for system_name in population_group:
                if system != '*' and system_name != system:
                    continue
                system_group = population_group[system_name]
                for









#########################" OLD CODE "#########################
# import h5py
# from PBSHM_mdof.system.population import Population
# import re

# class HDF5Handler:
#     def __init__(self, file: h5py.File):
#         self.file = file
        
#     def save_simulation_data(self,dt,t_end,population:Population, 
#     simulation_data: dict,id_simulation:int,latent_value:float,resonance_frequency:dict):
#         """
#         Save simulation data to the HDF5 file.
#         simulation_data should be a dictionary with the following structure:
#             {
#                 'time': time_data,
#                 'population': {
#                     'system_1': {
#                         'simulation_1': {
#                             attrs: {temperature: temperature}
                            
#                             'output': output_data,
#                             'input': input_data
#                         },
#                         'simulation_2': {...}
#                     },
#                     'system_2': {...}
#                 }
#             }
#         """

#         population_name = '_'.join([population.name,str(population.state),str(population.anomaly_level)])
#         population_ds = self.file.require_group(population_name)
#         population_ds.attrs['dt'] = dt
#         population_ds.attrs['t_end'] = t_end

#         population_ds.attrs['state'] = population.state
#         population_ds.attrs['anomaly_level'] = population.anomaly_level
#         population_ds.attrs['resonance_frequency'] = list(resonance_frequency.values())

#         for sys_name, sys_data in population.systems_params.items():
#             system = population_ds.require_group(sys_name)
#             system.attrs['mass'] = sys_data['mass']
#             system.attrs['stiffness'] = sys_data['stiffness']
#             system.attrs['damping'] = sys_data['damping']
#             system.attrs['units'] = 'kg, N/m, Ns/m'
        
#         for sys_name, sys_data in simulation_data.items():
            
#             simulation_ds = population_ds[sys_name].create_group(f'simulation_{id_simulation}')
#             simulation_ds.attrs['latent_value'] = latent_value
#             if 'output' not in simulation_ds:
#                 simulation_ds.create_dataset('output', shape=sys_data['output'].shape, dtype=sys_data['output'].dtype, data=sys_data['output'])
#             if 'input' not in simulation_ds:
#                 simulation_ds.create_dataset('input', shape=sys_data['input'].shape, dtype=sys_data['input'].dtype, data=sys_data['input'])
#     def get_data_iterator(self):
#         for population_name in self.file:
#             population_group = self.file[population_name]
#             for system_name in population_group:
#                 system_group = population_group[system_name]
#                 for simulation_name in system_group:
#                     simulation_group = system_group[simulation_name]
#                     if 'output' in simulation_group:
#                         res = {'output':simulation_group['output'][:],
#                         'dt':population_group.attrs['dt'], 
#                         'system_name':system_name, 
#                         'state':population_group.attrs['state'], 
#                         'anomaly_severity':population_group.attrs['anomaly_level'],
#                         'latent_value':simulation_group.attrs['latent_value'],
#                         'resonance_frequency':population_group.attrs['resonance_frequency']}
    
#                         yield res

#     def get_data(self, population:str='*', system:str='*', simulation:str='*'):
#         """
#         Get the output data, system name, state, and anomaly_severity for all the simulations in the HDF5 file.
#         Returns:
#             A list of dictionaries, where each dictionary contains the following information for a single simulation:
#                 {'system_name': system_name, 'state': state, 'anomaly_severity': anomaly_severity, 'output': output_data}
#         """
#         simulation_data = []
#         population_data = self.file[population]
#         for sys_name in population_data:
#             if system != '*' and sys_name != system:
#                 continue
#             system_group = population_data[sys_name]
#             for sim_name in system_group:
#                 if simulation != '*' and sim_name != simulation:
#                     continue
#                 simulation_group = system_group[sim_name]
#                 output_data = simulation_group['output'][:]
#                 state = population_data.attrs['state']
#                 anomaly_severity = population_data.attrs['anomaly_level']
#                 simulation_data.append({'system_name': sys_name, 'state': state, 'anomaly_level': anomaly_severity, 'output': output_data})
#         time = {'dt':population_data.attrs['dt'] , 't_end':population_data.attrs['t_end']}

#         return simulation_data , time

    


#     def close(self):
#         self.file.close()