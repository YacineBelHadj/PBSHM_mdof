#%%
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
from config import settings
import matplotlib.pyplot as plt
import numpy as np
import logging
import seaborn as sns
from sklearn.model_selection import train_test_split


logging_path = Path(settings.default['path']['abspath']) / 'logs'/ 'formating_data.log'
logging.basicConfig(level=logging.INFO, 
format='%(asctime)s - %(levelname)s - %(message)s',
filemode='w',filename=logging_path)

# set path to parquet file
path_processed_data = Path(settings.default['path']['abspath']) / 'data' / 'processed4' / 'data.parquet'

# read parquet file into PyArrow table
table = pq.read_table(str(path_processed_data))
df = table.to_pandas()
freqs = df['freqs'][0]

# Check for NaN values in PSDs
if df['psd'].isna().sum()==0:
    logging.info('No NaN values in PSDs')
else:
    logging.error('NaN values in PSDs')
#%%

index_health = np.where(df['state']=='healthy')[0]
#%%
# Split data into train and test
index_train, index_test = train_test_split(index_health,
                                            test_size=0.3,
                                            random_state=42, 
                                            shuffle=True,
                                            stratify=df.iloc[index_health]['system_name'].values)

fig, ax = plt.subplots()

ax.hist(df.iloc[index_train]['system_name'], bins=range(21), rwidth=0.8)
ax.set_xlabel('System ID')
ax.set_ylabel('Count')
ax.set_xticklabels([int(label.get_text().split('_')[1]) for label in ax.get_xticklabels()])
plt.title('Train set distribution')
plt.show()
plt.close()
#%%  
index_healthy_system_0 = np.where(df.iloc[index_train]['system_name']=='system_0')[0]
fig, ax = plt.subplots()
for psd in df['psd'][index_healthy_system_0][:20]:
    ax.plot(psd, alpha=0.1,color='blue')
ax.set_title('PSD of healthy system 0')
ax.set_yscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD')
plt.show()
plt.close()
#%%


