import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
from config import settings
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from PBSHM_mdof.visualization.visualize import plot_psd_example
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)

def format_data(test_size=0.3,data_name='no_noise.parquet',verbose=False):
    """This function formats the data for the training of the classifier.
    It reads the parquet file, splits the data into train and test set and
    returns the train and test set as PyArrow tables.
    """
    # set path to parquet file
    path_processed_data = Path(settings.default['path']['abspath']) / 'data' / 'processed_psd_new' / data_name

    # read parquet file into PyArrow table
    table = pq.read_table(str(path_processed_data))
    df = table.to_pandas()
    df['latent_value']=df['latent_value'].astype('float32')
    # Check for NaN values in PSDs
    if df['psd'].isna().sum()==0:
        logging.info('No NaN values in PSDs')
    else:
        logging.error('NaN values in PSDs')
    
    index_health = np.where(df['state']=='healthy')[0]
    
    # Split data into train and test
    index_train, index_test = train_test_split(index_health,
                                                test_size=test_size,
                                                random_state=42, 
                                                shuffle=True,
                                                stratify=df.iloc[index_health]['system_name'].values)
    
    # compute the counts for each system name in the train set
    counts = df.iloc[index_train]['system_name'].value_counts()
    
    # log the counts for each system name
    for system, count in counts.items():
        logging.info(f"Number of experiments in train set for {system}: {count}")
    
    # plot PSDs of system_0 in train set
    index_healthy_system_0 = np.where(df.iloc[index_train]['system_name']=='system_0')[0]
    if verbose:
        plot_psd_example(df.iloc[index_healthy_system_0[:10]]['psd'].values)
        plt.show()
        plt.close()
    # normalize PSDs
    df['psd']=df['psd'].apply(lambda x: np.log10(x))
    min_data = np.stack(df['psd'].values).min()
    max_data = np.stack(df['psd'].values).max()
    df['psd']=df['psd'].apply(lambda x: (x-min_data)/(max_data-min_data))

    #encode labels 
    le = OneHotEncoder(handle_unknown='ignore')
    labels = le.fit_transform(df['system_name'].values.flatten().reshape(-1,1)).toarray()
    transformer = {'label_encder': le,
                   'normalizer': {'min': min_data, 'max': max_data}}
    return df, labels, index_train, index_test, transformer
