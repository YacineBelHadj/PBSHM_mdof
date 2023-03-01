from PBSHM_mdof.data.data_handler import HDF5DataBuilder
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import h5py
from pathlib import Path
from scipy.signal import welch
from tqdm import tqdm
from config import settings

def compute_PSD(signal_data, dt, nperseg=1024):
    f, psd = welch(signal_data, 1/dt, nperseg=nperseg, scaling='spectrum')
    return f, psd

def main():
    # Create a schema for the PyArrow Table

    # open the original HDF5 file
    path_generated_dataset = Path(settings.default['path']['abspath']) / Path(settings.default['path']['generated_dataset'])
    path_saved_data = Path(settings.default['path']['abspath']) / 'data' / 'processed4' / 'data.parquet'

    result_list = []  # create an empty list to hold the result dictionaries

    with h5py.File(path_generated_dataset, 'r') as input_file:
        dh = HDF5DataBuilder(input_file)
        dt = dh.get_simulation_dt('POPULATION_1', 'default_simulation')
        iterator = dh.get_data_iterator()

        # loop over the data and compute the PSDs
        for d in tqdm(iterator):
            tdd = d['TDD']
            for k, v in tdd.items():
                acc_7 = v[:, 2 * 8 + 1]
                f, psd = compute_PSD(acc_7, dt)
                system_name = k
                anomaly_level = d['experiment_params']['anomaly_level']
                state = d['experiment_params']['state']
                latent_value = d['experiment_params']['latent_value']
                fr = d['resonance_frequency_data'][k]


                # save the data
                result_dict = {'system_name':system_name,
                                'psd': psd, 
                                'resonance_freq': fr, 
                                'anomaly_level': anomaly_level,
                                'state': state, 
                                'latent_value': latent_value}
                result_list.append(result_dict)  # append the result dict to the list

    # convert the list of result dicts to a PyArrow Table and save to Parquet file
    result_table = pa.Table.from_pandas(pd.DataFrame(result_list))
    pq.write_table(result_table, str(path_saved_data))

if __name__ == '__main__':
    main()
