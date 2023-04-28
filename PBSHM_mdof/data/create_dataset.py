from data_handler import HDF5DataBuilder
from PBSHM_mdof.system.simulation import Simulation
from PBSHM_mdof.system.population import Population
from PBSHM_mdof.system.mdof_system import MdofSystem
from PBSHM_mdof.system.population_manipulator import PopulationManipulator
from utils import resonance_frequency_computation

from config import settings
import numpy as np
from tqdm import tqdm
import h5py
import logging

logging.basicConfig(level=logging.INFO, 
format='%(asctime)s - %(levelname)s - %(message)s',
filemode='w',filename='logs/generate_data.log')

dt = settings.default['simulation']['dt'] 
t_end = settings.default['simulation']['t_end']
path_dataset = settings.default['path']['generated_dataset']
population_params_path= settings.default['path']['population_params']

def main():


    population = Population()
    population.load_population(population_params_path)
    logging.info(f'Loaded population from file {population_params_path}')
    pop_manipulator = PopulationManipulator(population)

    with h5py.File(path_dataset, 'w') as f:
        dh = HDF5DataBuilder(f)
        
        # Save healthy simulations and population parameters
        pop_grp = dh.set_save_population(population=population)
        std_latent = 20
        mean_latent = 50
        simulation_name='default_simulation'
        simu_grp = dh.set_save_simulation_params(pop_grp,simulation_name, dt=dt, t_end=t_end, std_latent=std_latent)

        # run experiments
        state = 'healthy'
        for i in tqdm(range(1200)):
            # parameter for the simulation
            amplitude = 5 * (-np.log(1-np.random.uniform(0, 1 )))**(1/1.9)+10
            anomaly_level=0
            loc = 7
            latent_value = np.random.normal(mean_latent, std_latent)
            experiment_name = f'experiment_{i}_{state}_{anomaly_level}'

            # do experiments

            
            attempt = 1
            run =True
            while run : 
                requests = [{'type': 'environment', 'latent_value': latent_value, 'coefficients': 'load'}]
                population_affected = pop_manipulator.affect(requests)
                simulator = Simulation(population_affected, dt, t_end)
                simulation_data = simulator.simulation_white_noise(location=loc, amplitude=amplitude)
                if simulation_data is None:
                    latent_value = np.random.normal(mean_latent, std_latent)
                    amplitude = np.sqrt(np.square(np.random.normal(15, 5)) + 10)
                    logging.info(f'Attempt {attempt} for simulation {i}')
                    attempt += 1

                else:
                    run=False 

            
            # save experiments
            exp_grp = dh.set_save_experiment_params(simu_grp, experiment_name, latent_value, anomaly_level, state, loc, amplitude)
            dh.save_experiment_population_params(exp_grp, population_affected)
            dh.save_experiment_time_domain_data(exp_grp, simulation_data)
            
            logging.info(f'Saved simulation data for id={i}, latent_value={latent_value}')
        
        # Save anomalous simulations
        for anomaly_level in range(1, 14, 2):

            state = 'anomalous'            
            for i in tqdm(range(200)):
                # parameter for the simulation
                ai = anomaly_level/100
                amplitude = 5 * (-np.log(1-np.random.uniform(0, 1)))**(1/1.9)+10
                loc = 7
                latent_value = np.random.normal(mean_latent, std_latent)
                experiment_name = f'experiment_{i}_{state}_{ai}'
                # do experiments
                attempt = 1
                run =True
                while run : 
                    requests = [{'type': 'environment', 'latent_value': latent_value, 'coefficients': 'load'},
                            {'type': 'anomaly', 'location': 5, 'anomaly_size': ai, 'anomaly_type': 'stiffness'}]
                    population_affected = pop_manipulator.affect(requests)
                    simulator = Simulation(population_affected, dt, t_end)
                    simulation_data = simulator.simulation_white_noise(location=loc, amplitude=amplitude)
                    if simulation_data is None:
                        latent_value = np.random.normal(mean_latent, std_latent)
                        amplitude = np.sqrt(np.square(np.random.normal(15, 5)) + 10)
                        logging.info(f'Attempt {attempt} for simulation {i}')
                        attempt += 1

                    else:
                        run=False 

                
                exp_grp = dh.set_save_experiment_params(simu_grp, experiment_name, latent_value, ai, state, loc, amplitude)
                dh.save_experiment_population_params(exp_grp, population_affected)
                dh.save_experiment_time_domain_data(exp_grp, simulation_data)
                
                logging.info(f'Saved simulation data for id={i},anomaly_level={ai}, latent_values={latent_value}')

if __name__=='__main__':
    main()