import numpy as np
from scipy import signal
import h5py
from .mdof_system import MdofSystem
from .population import Population

class Simulation:
    """
    """
    def __init__(self, population: Population, dt: float,t_end:float):
        self.population = population
        self.dt = dt
        self.t_end = t_end

    def simulation_white_noise(self,location: int = 7,amplitude: float = 300):
        t = np.arange(0, self.t_end, self.dt)
        data = {}
        
        for sys_name,sys_param in self.population.systems_matrices.items():
            sys = MdofSystem(**sys_param)
            u, (t_out_,y,x_)=sys.simulate_white_noise(t=t,location=location,amplitude=amplitude)

            data[sys_name] = {'time':t,'input':u,'output':y}
        return data

    def simulation_white_noise_tf(self,i: int = 7,j:int=1,amplitude: float = 300):
        omega = np.arange(0, 1/self.dt, 1/self.t_end)
        freq = omega/(2*np.pi)
        data = {}
        for sys_name,sys_param in self.population.systems_matrices.items():
            sys = MdofSystem(**sys_param)
            tf = sys.transfer_function(omega,i,j)
            u = np.random.normal(0,amplitude,(1,))
            u_f = np.fft.fft(u)

            y = tf*u_f
            data[sys_name] = {'freq':freq,'tsf':y}
        raise NotImplementedError
        return data


        
    def simulate(self,u):
        t = np.arange(0, self.t_end, self.dt)
        assert len(u)==len(t) 
        data = {}
        
        for sys_name,sys_param in self.population.systems_matrices.items():
            sys = MdofSystem(**sys_param)
            t_out_,y,x_=sys.simulate_lsim(u[sys_name],t=t,)
            data[sys_name] = {'time':t,'input':u,'output':y}
        return data


