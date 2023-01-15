import configparser
import numpy as np 
from pathlib import Path 
import ast

def load_config_file(dir_config:Path=None):
    if dir_config is None:
        dir_config = Path(__file__).parent/'config_population.ini' 
    config = configparser.ConfigParser()
    config.read(dir_config)
    return config

def correct_form(lst):
    return np.array(list(map(float ,lst[1:-1].split(','))))

def extract_variables(config: configparser.ConfigParser, population_name:str ='POPULATION_1'):
    data_dict = config[population_name]

    N = int(data_dict['N'])
    N_dof = int(data_dict['N_dof'])
    m_mean = np.array(correct_form(data_dict['m_mean']))
    m_std = float(data_dict['m_std'])
    k_mean = np.array(correct_form(data_dict['k_mean']))
    k_std = float(data_dict['k_std'])
    c_mean = np.array(correct_form(data_dict['c_mean']))
    c_std = float(data_dict['c_std'])

    return N, N_dof, m_mean, m_std, k_mean, k_std, c_mean, c_std



def save_config_file(dir_config:Path=None):
    if dir_config is None:
        dir_config = Path(__file__).parent/'config_population.ini' 
    config = configparser.ConfigParser()
    config['POPULATION_1'] = {
            'N':20, 
            'N_dof' : 8, 
            'm_mean': np.array([0.5318, 0.4040, 0.4101, 0.4123, 0.3960, 0.3809, 0.4086, 0.3798]).tolist(),
            'k_mean': (1e3*np.array([1e-6, 56.70, 56.70, 56.70, 56.70, 56.70, 56.70, 56.70])).tolist(),
            'c_mean': np.array([8.746, 8.791, 8.801, 8.851, 8.714, 8.737, 8.549, 8.752]).tolist(),
            'm_std': 0.03, 
            'k_std': 0.01, 
            'c_std': 0.08
    }
    with open(dir_config, 'w') as configfile:
        config.write(configfile)



if __name__ == '__main__':
    save_config_file()
    conf= load_config_file()['POPULATION_1']
    print(correct_form(conf['m_mean']))
