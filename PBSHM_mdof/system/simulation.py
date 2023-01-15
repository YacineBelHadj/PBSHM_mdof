import numpy as np
from scipy import signal
import h5py
from .mdof_system import Mdof_system
from .population import Population

class Simulation:
    def __init__(self, population: Population, dt: float,t_end:float):
        self.population = population
        self.dt = dt
        self.t_end = t_end

    def generate_simulation_data(self):
        t = np.arange(0, self.t_end, self.dt)
        data = {}
        
        for sys_name,sys_param in self.population.systems_matrices.items():
            sys = Mdof_system(*sys_param)
            u, (t_out_,y,x_)=sys.simulate_white_noise(t=t)
            data[sys_name] = {'time':t,'input':u,'output':y}
        return data


