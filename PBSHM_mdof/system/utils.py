import pickle

def save_system_params(system_params: dict, filepath: str):
    with open(filepath, 'wb') as f:
        pickle.dump(system_params, f)

def load_system_params(filepath: str):
    with open(filepath, 'rb') as f:
        system_params = pickle.load(f)
    return system_params