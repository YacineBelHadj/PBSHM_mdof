from __future__ import annotations
from config import settings
from dataclasses import dataclass
from PBSHM_mdof.system.population import Population
from PBSHM_mdof.system.simulation import Simulation
from PBSHM_mdof.system.mdof_system import MdofSystem
from copy import deepcopy
import numpy as np
from typing import Optional, Union, Any
from pathlib import Path

import pickle
from dataclasses import dataclass

from typing import Optional
from copy import deepcopy

from PBSHM_mdof.system.population import Population
import numpy as np

class Handler:
    def __init__(self, next_handler=None):
        self.next_handler = next_handler
    
    def set_next(self, handler: Handler) -> Handler:
        self.next_handler = handler
        return handler
    
    def handle(self, population: Population, request) -> Optional[Population]:
        if self.next_handler is not None:
            return self.next_handler.handle(population, request)
        return None


class EnvironmentHandler(Handler):
    def __init__(self, next_handler=None, population_coef_latent=None):
        super().__init__(next_handler)
        self.population_coef_latent = population_coef_latent

    def handle(self, population: Population, request) -> Optional[Population]:
        if request['type'] == 'environment':
            if request['coefficients'] == 'generate':
                coef = np.array(settings.default['latent_coef'])/10

                self.population_coef_latent = {f'system_{i}': coef *
                                                              np.random.normal(1, 1e-2, size=8) for i in range(20)}
                write_population_coef_latent(self.population_coef_latent)
            
            if request['coefficients'] == 'load':
                self.population_coef_latent = read_population_coef_latent()

            latent_delta = request['latent_value']
            for sys_name, sys_param in population.systems_params.items():
                sys_param['stiffness'] = sys_param['stiffness'] - latent_delta * self.population_coef_latent[sys_name]

            return population

        return super().handle(population, request)


class AnomalyHandler(Handler):
    def handle(self, population: Population, request) -> Optional[Population]:
        if request['type'] == 'anomaly':
            location = request['location']
            anomaly_type = request['anomaly_type']
            anomaly_size = request['anomaly_size']
            # Create a new Population object with a copy of the current object's properties
            new_population = Population(population.systems_params.copy())
            new_population.anomaly_level = anomaly_size
            new_population.state = 'anomalous'
            if anomaly_type not in ['stiffness', 'mass', 'damping']:
                raise ValueError(f"Invalid anomaly type. Choose from {new_population.AVAILABLE_ANOMALY_TYPES.keys()}.")
            # Modify the systems in the new object
            for key, values in new_population.systems_params.items():
                arr = np.copy(new_population.systems_params[key][anomaly_type])
                arr[location] *= (1 - anomaly_size)
                new_population.systems_params[key][anomaly_type] = arr
            return new_population
        return super().handle(population, request)


class PopulationManipulator:
    def __init__(self, population: Population):
        self.population = population
        self.handlers = [EnvironmentHandler(), AnomalyHandler()]

    def affect(self, requests, inplace: bool = False) -> Optional[Population]:
        population_to_update = self.population if inplace else deepcopy(self.population)
        for request in requests:
            for handler in self.handlers:
                new_population = handler.handle(population_to_update, request)
                if new_population is not None:
                    population_to_update = new_population
        population_to_update.compute_systems_matrices()
        if inplace:
            return population_to_update
        else:
            return deepcopy(population_to_update)

def read_population_coef_latent():
    with open(Path(__file__).parent.parent.parent /'data/systems/systems_latent_coef.pkl','rb') as f:
        population_coef_latent = pickle.load(f)
    return population_coef_latent

def write_population_coef_latent(population_coef_latent):
    with open(Path(__file__).parent.parent.parent /'data/systems/systems_latent_coef.pkl', 'wb') as f:
        pickle.dump(population_coef_latent, f)

import matplotlib.pyplot as plt

if __name__ == "__main__":
    population = Population()
    population.generate_population()
    manipulator = PopulationManipulator(population)
    requests = [{'type': 'anomaly', 'location': 5, 'anomaly_size': 0, 'anomaly_type': 'stiffness'},
        {'type': 'environment', 'latent_value': 40, 'coefficients': 'load'}]
    requests2=[{'type': 'anomaly', 'location': 5, 'anomaly_size': 0.5, 'anomaly_type': 'stiffness'},
        {'type': 'environment', 'latent_value': 0, 'coefficients': 'load'}]

    manipulated_population = manipulator.affect(requests)
    manipulated_population2 = manipulator.affect(requests2)
    omega=np.linspace(0,1000,1000)
    freq = omega/(2*np.pi)
    h_h = np.abs(MdofSystem(**(population.systems_matrices["system_0"])).transfer_function(omega,1,7))
    h_m=np.abs(MdofSystem(**(manipulated_population.systems_matrices["system_0"])).transfer_function(omega,1,7))
    h_m2=np.abs(MdofSystem(**(manipulated_population2.systems_matrices["system_0"])).transfer_function(omega,1,7))


    plt.plot(freq[100:1300],h_h[100:1300],label='healthy')
    plt.plot(freq[100:1300],h_m[100:1300],label='latent')
    plt.plot(freq[100:1300],h_m2[100:1300],label='anomaly')
    plt.yscale('log')
    plt.legend()
    plt.show()
    plt.close()

# from flask import Flask, render_template, request, jsonify
# import numpy as np
# from matplotlib.figure import Figure
# from io import BytesIO
# import base64

# app = Flask(__name__)

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/plot", methods=["POST"])
# def plot():
#     latent_value = float(request.form["latent_value"])
#     anomaly_size = float(request.form["anomaly_size"])

#     population = Population()
#     population.generate_population()
#     manipulator = PopulationManipulator(population)
#     requests = [
#         {"type": "anomaly", "location": 5, "anomaly_size": anomaly_size, "anomaly_type": "stiffness"},
#         {"type": "environment", "latent_value": latent_value, "coefficients": "load"}
#     ]
#     manipulated_population = manipulator.affect(requests)

#     omega = np.linspace(0, 1000, 1000)
#     freq = omega / (2 * np.pi)
#     h_h = np.abs(Mdof_system(**(population.systems_matrices["system_0"])).transfer_function(omega, 1, 7))
#     h_m = np.abs(Mdof_system(**(manipulated_population.systems_matrices["system_0"])).transfer_function(omega, 1, 7))

#     fig = Figure()
#     ax = fig.add_subplot(1, 1, 1)
#     ax.plot(freq[100:1300], h_h[100:1300], label="healthy")
#     ax.plot(freq[100:1300], h_m[100:1300], label="latent")
#     ax.set_yscale("log")
#     ax.legend()

#     buffer = BytesIO()
#     fig.savefig(buffer, format='png')
#     buffer.seek(0)

#     data_uri = base64.b64encode(buffer.read()).decode('ascii')
#     return data_uri

# if __name__ == "__main__":
#     app.run(debug=True)


# @dataclass
# class Population_EOC_generator:
#     population : Population
#     population_coef_latent : dict | None = None

#     def __call__(self, latent :float) -> Population:
#         return self.affect_latent(latent)

#     def generate_coefficients(self):
#         coef = [1e-8,-200,-200,3575,-300,-3500,+1000,-450]
#         coef *= np.random.normal(1,0.05,size=8)
#         coef = np.array(coef)/10
#         self.population_coef_latent = {f'system_{i}':coef*np.random.normal(1,1e-2,size=8) for i in range(20)}

#     def affect_latent(self,latent:float):
#         return affect_latent(self.population,self.population_coef_latent,latent)

#     def coefficent_save(self,coefficent_path):
#         with open(coefficent_path,'wb') as f:
#             pickle.dump(self.population_coef_latent,f)

#     def coefficent_load(self,coefficent_path):
#         with open(coefficent_path,'rb') as f:
#             self.population_coef_latent = pickle.load(f)



    
