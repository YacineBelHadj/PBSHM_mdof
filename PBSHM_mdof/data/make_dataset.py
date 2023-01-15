from PBSHM_mdof.data.data_handler import HDF5Handler
from PBSHM_mdof.system.simulation import Simulation
from PBSHM_mdof.system.population import Population
import h5py
import numpy as np
from tqdm import tqdm
from copy import deepcopy
# Create a new HDF5 file
def main(new_population:bool=False,dt:float=0.0025,t_end:float=30.0):
    population = Population()
    if new_population:
        population.generate_system_variables()
        population.save_population('data/systems/systems_healthy.json')
        print('New data generated and saved to data/systems/systems_healthy.json')
    else: 
        population.load_population('data//systems/systems_healthy.json')
        print('Loaded data from data/systems/systems_healthy.json')
    if True:
        with h5py.File('data/raw/datatest.hdf5','w') as f:
            datahandler = HDF5Handler(f)
            simulator = Simulation(population,dt,t_end)
            for id in tqdm(range(1200)):
                simulation_data = simulator.generate_simulation_data()
                datahandler.save_simulation_data(dt=dt,t_end=t_end,
                population=population,simulation_data=simulation_data,id_simulation=id)
    with h5py.File('data/raw/datatest.hdf5','a') as f:
        datahandler = HDF5Handler(f)
              
        for anomaly_level in range(1,14,2):
            population_ano = deepcopy(population)
            population_ano.add_anomaly(anomaly_size=anomaly_level/100)
            simulator = Simulation(population_ano,dt,t_end)  
            for id in tqdm(range(200)):
                simulation_data = simulator.generate_simulation_data()
                datahandler.save_simulation_data(dt=dt,t_end=t_end,
                population=population_ano,simulation_data=simulation_data,id_simulation=id)




if __name__=='__main__':
    main()

