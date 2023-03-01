from PBSHM_mdof.data.data_handler import HDF5Handler
from PBSHM_mdof.system.simulation import Simulation
from PBSHM_mdof.system.population import Population
from PBSHM_mdof.system.mdof_system import MdofSystem
from PBSHM_mdof.system.population_manipulator import PopulationManipulator
from config import settings
import h5py
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import logging

logging.basicConfig(level=logging.INFO, 
format='%(asctime)s - %(levelname)s - %(message)s',
filemode='w',filename='generate_data.log')

def resonance_frequency(population:Population):
    res_freq = {}
    for sys_name,sys_param in population.systems_matrices.items():
        sys = MdofSystem(**sys_param)
        res_freq[sys_name] = sys.resonance_frequency()
    return res_freq

def main(new_population=True, dt=0.0025, t_end=10.0):
    population = Population()
    population_params= settings.default['path']['population_params']
    if new_population:
        population.generate_population()
        population.save_population(population_params)
        logging.info(f'New data generated and saved to file {population_params}')
    else: 
        population.load_population(population_params)
        logging.info(f'Loaded data from file {population_params}')
    pop_manip = PopulationManipulator(population)
    if True:
        with h5py.File('data/raw/datatest.hdf5','w') as f:
            datahandler = HDF5Handler(f)
            for id in tqdm(range(1200)):
                latent_value = np.random.normal(50,30)
                requests = [{'type': 'environment', 'latent_value': latent_value, 'coefficients': 'load'}]
                population_affected = pop_manip.affect(requests)
                res_freq = resonance_frequency(population_affected)
                simulator = Simulation(population_affected,dt,t_end)
                amplitude = np.square(np.random.normal(1,10))+1
                loc = 7
                simulation_data = simulator.simulation_white_noise(location=loc,amplitude=amplitude)
                datahandler.save_simulation_data(dt=dt,t_end=t_end,
                    population=population,simulation_data=simulation_data,
                    id_simulation=id,latent_value=latent_value,
                    resonance_frequency=res_freq)
                logging.info(f'Saved simulation data for id={id}, latent_value={latent_value}')

    with h5py.File('data/raw/datatest.hdf5','a') as f:
        datahandler = HDF5Handler(f)
              
        for anomaly_level in range(1,14,2):

            for id in tqdm(range(200)):
                latent_values = np.random.normal(50,30)
                requests = [{'type': 'anomaly', 'location': 5, 'anomaly_size': anomaly_level/100, 'anomaly_type': 'stiffness'},
                    {'type': 'environment', 'latent_value': latent_values, 'coefficients': 'load'}]

                population_affected = pop_manip.affect(requests)
                res_freq = resonance_frequency(population_affected)
                simulator = Simulation(population_affected,dt,t_end)  
                amplitude = np.square(np.random.normal(50,100))+1
                loc = 7
                simulation_data = simulator.simulation_white_noise(location=loc,amplitude=amplitude)  
                datahandler.save_simulation_data(dt=dt,t_end=t_end,
                    population=population_affected,simulation_data=simulation_data,
                    id_simulation=id,latent_value=latent_values,resonance_frequency=res_freq)
                
                logging.info(f'Saved simulation data for id={id}, anomaly_level={anomaly_level/100}, latent_values={latent_values}')


if __name__=='__main__':
    main()
