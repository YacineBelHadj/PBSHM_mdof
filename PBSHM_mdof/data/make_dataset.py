from PBSHM_mdof.data.data_handler import HDF5Handler
from PBSHM_mdof.system.simulation import Simulation
from PBSHM_mdof.system.population import Population
from PBSHM_mdof.system.population_manipulator import PopulationManipulator
import h5py
import numpy as np
from tqdm import tqdm
from copy import deepcopy
# Create a new HDF5 file

def main(new_population:bool=True,dt:float=0.0025,t_end:float=30.0):
    population = Population()
    if new_population:
        population.generate_population()
        population.save_population('data/systems/systems_healthy.json')
        print('New data generated and saved to data/systems/systems_healthy.json')
    else: 
        population.load_population('data/systems/systems_healthy.json')
        print('Loaded data from data/systems/systems_healthy.json')
    pop_manip = PopulationManipulator(population)

    if True:
        with h5py.File('data/raw/datatest.hdf5','w') as f:
            datahandler = HDF5Handler(f)
            for id in tqdm(range(1200)):

                latent_value = np.random.normal(50,30)
                requests = [{'type': 'environment', 'latent_value': latent_value, 'coefficients': 'load'}]
                population_affected = pop_manip.affect(requests)
                simulator = Simulation(population_affected,dt,t_end)
                simulation_data = simulator.simulation_white_noise()
                datahandler.save_simulation_data(dt=dt,t_end=t_end,
                population=population,simulation_data=simulation_data,id_simulation=id,latent_value=latent_value)
    with h5py.File('data/raw/datatest.hdf5','a') as f:
        datahandler = HDF5Handler(f)
              
        for anomaly_level in range(1,14,2):
            latent_values = np.random.normal(50,30)
            requests = [{'type': 'anomaly', 'location': 5, 'anomaly_size': anomaly_level/100, 'anomaly_type': 'stiffness'},
                {'type': 'environment', 'latent_value': latent_values, 'coefficients': 'load'}]

            population_affected = pop_manip.affect(requests)
            simulator = Simulation(population_affected,dt,t_end)  
            for id in tqdm(range(200)):
                simulation_data = simulator.simulation_white_noise()
                datahandler.save_simulation_data(dt=dt,t_end=t_end,
                population=population_affected,simulation_data=simulation_data,id_simulation=id,latent_value=latent_value)




if __name__=='__main__':
    main()
