import h5py
from PBSHM_mdof.system.population import Population
import re

class HDF5Handler:
    def __init__(self, file: h5py.File):
        self.file = file
        
    def save_simulation_data(self,dt,t_end,population:Population, simulation_data: dict,id_simulation:int):
        """
        Save simulation data to the HDF5 file.
        simulation_data should be a dictionary with the following structure:
            {
                'time': time_data,
                'population': {
                    'system_1': {
                        'simulation_1': {
                            'output': output_data,
                            'input': input_data
                        },
                        'simulation_2': {...}
                    },
                    'system_2': {...}
                }
            }
        """

        population_name = '_'.join([population.name,str(population.state),str(population.anomaly_level)])
        population_ds = self.file.require_group(population_name)
        population_ds.attrs['dt'] = dt
        population_ds.attrs['t_end'] = t_end

        population_ds.attrs['state'] = population.state
        population_ds.attrs['anomaly_level'] = population.anomaly_level
        for sys_name, sys_data in population.systems_params.items():
            system = population_ds.require_group(sys_name)
            system.attrs['masses'] = sys_data[0]
            system.attrs['stiffness'] = sys_data[1]
            system.attrs['damping'] = sys_data[2]
            system.attrs['units'] = 'kg, N/m, Ns/m'

        for sys_name, sys_data in simulation_data.items():
            
            simulation_ds = population_ds[sys_name].create_group(f'simulation_{id_simulation}')
            if 'output' not in simulation_ds:
                simulation_ds.create_dataset('output', shape=sys_data['output'].shape, dtype=sys_data['output'].dtype, data=sys_data['output'])
            if 'input' not in simulation_ds:
                simulation_ds.create_dataset('input', shape=sys_data['input'].shape, dtype=sys_data['input'].dtype, data=sys_data['input'])
    
    def get_data_iterator(self, population:str='*', system:str='*', simulation:str='*'):
        """
        Iterate through the data in the HDF5 file, returning a tuple of 
        (system_name, state, anomaly_level, output_data) for each simulation that matches
        the specified population, system, and simulation filters.
        """
        population_data = self.file[population]
        for sys_name in population_data:
            if system != '*' and sys_name != system:
                continue
            system_group = population_data[sys_name]
            for sim_name in system_group:
                if simulation != '*' and sim_name != simulation:
                    continue
                simulation_group = system_group[sim_name]
                output_data = simulation_group['output'][:]
                state = population_data.attrs['state']
                anomaly_level = population_data.attrs['anomaly_level']
                yield (sys_name, state, anomaly_level, output_data)
    def get_data(self, population:str='*', system:str='*', simulation:str='*'):
        """
        Get the output data, system name, state, and anomaly_severity for all the simulations in the HDF5 file.
        Returns:
            A list of dictionaries, where each dictionary contains the following information for a single simulation:
                {'system_name': system_name, 'state': state, 'anomaly_severity': anomaly_severity, 'output': output_data}
        """
        simulation_data = []
        population_data = self.file[population]
        for sys_name in population_data:
            if system != '*' and sys_name != system:
                continue
            system_group = population_data[sys_name]
            for sim_name in system_group:
                if simulation != '*' and sim_name != simulation:
                    continue
                simulation_group = system_group[sim_name]
                output_data = simulation_group['output'][:]
                state = population_data.attrs['state']
                anomaly_severity = population_data.attrs['anomaly_level']
                simulation_data.append({'system_name': sys_name, 'state': state, 'anomaly_level': anomaly_severity, 'output': output_data})
        time = {'dt':population_data.attrs['dt'] , 't_end':population_data.attrs['t_end']}

        return simulation_data , time

    


    def close(self):
        self.file.close()