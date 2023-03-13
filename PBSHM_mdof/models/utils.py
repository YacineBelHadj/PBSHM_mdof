from sklearn.metrics import roc_auc_score
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm



def compute_auc_df(df):
    system_name = df['system_name'].unique().tolist()
    system_name.sort(key=lambda x: int(x.split('_')[-1]))
    anomaly_level = np.sort(df['anomaly_level'].unique())

    auc_df = pd.DataFrame(columns=system_name)
    for system_id in system_name:
        healthy_training_data = df['confidence'][(df['system_name'] == system_id) & (df['type'] == 'train')]
        healthy_testing_data = df['confidence'][(df['system_name'] == system_id) & (df['type'] == 'test') & (df['anomaly_level'] == '0')]
        for al in anomaly_level:
            if al == '0':
                auc = compute_auc(healthy_training_data, healthy_testing_data)
            else:
                anomaly_data = df['confidence'][(df['system_name'] == system_id) & (df['type'] == 'test') & (df['anomaly_level'] == al)]
                auc = compute_auc(healthy_testing_data, anomaly_data)
            auc_df.loc[f'auc_{al}', system_id] = auc
    return auc_df
# %%
def compute_auc(data_normal, data_anomaly):
    """
    Compute the AUC score for a given set of normal and anomaly data.

    Args:
    - data_normal: a dictionary containing the confidence scores of normal data
                   for a given system ID, with keys 'confidence'
    - data_anomaly: a dictionary containing the confidence scores of anomaly data
                    for a given system ID and anomaly level, with keys 'confidence'
    
    Returns:
    - auc_score: the AUC score
    """
    confidence_anomaly_0 = data_normal
    confidence_rest_of_anomalies = data_anomaly

    # create the labels for the data
    # 1 for anomaly 0 and 0 for the rest of anomalies
    labels_anomaly_0 = [1] * len(confidence_anomaly_0)
    labels_rest_of_anomalies = [0] * len(confidence_rest_of_anomalies)

    # concatenate the data and labels
    confidence = np.concatenate((confidence_anomaly_0, confidence_rest_of_anomalies))
    labels = np.concatenate((labels_anomaly_0, labels_rest_of_anomalies))
    # compute the AUC
    auc_score = roc_auc_score(labels, confidence)

    return auc_score


def plot_control_chart(df,system_id, column='confidence', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    data_system = df[df['system_name'] == system_id]
    df_train = data_system[data_system['type'] == 'train']
    df_test = data_system[data_system['type'] == 'test']
    df_test= df_test.sort_values(by='anomaly_level', ascending=True)

    df_plot = pd.concat([df_train, df_test])
    df_plot.reset_index(inplace=True)
    ax.plot(df_plot.index, df_plot[column], marker='.', linestyle='')
    anomaly_level=df_plot[df_plot['anomaly_level'].astype(float).diff()!=0]['anomaly_level'].to_dict()

    y_lowbound, y_upbound= ax.get_ylim()
    middle = (y_lowbound + y_upbound) / 2
    range = (y_upbound - y_lowbound) / 2
    text_y = middle - range * 0.5


    ax.axvline(x=len(df_train), color='firebrick', linestyle='--', alpha=0.5)
    ax.text(len(df_train), text_y, 'test data', rotation=90, color='firebrick', alpha=0.9)
    for key in list(anomaly_level.keys())[1:]:
        ax.axvline(x=key, color='firebrick', linestyle='--', alpha=0.5)
        ax.text(key, text_y, anomaly_level[key], color='steelblue', alpha=1)

    mean = np.mean(df_train[column])
    std = np.std(df_train[column])
    lcl = mean - 2 * std
    ax.plot()
    ax.axhline(y=lcl, color='firebrick', linestyle='--', alpha=0.5)
    ax.text(0, lcl-0.1*range, 'LCL', rotation=0, color='firebrick', alpha=0.9)
    # coloring out of control data with red 
    allert_df = df_plot[df_plot[column] < lcl]
    ax.plot(allert_df.index, allert_df[column], marker='.', linestyle='', color='red')

    ax.set_ylabel(column)
    ax.set_xlabel('index')
    ax.set_title(system_id)
    return fig, ax

def plot_control_chart_latent(system_id, result, column='confidence', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    data = result[system_id]
    data_level = pd.DataFrame(data)
    data_level.sort_values(by='anomaly level', inplace=True)
    data_level.reset_index(inplace=True)
    levels = np.unique(data_level['anomaly level'])
    
    # add color based on latent value
    colors = data_level['latent_value']
    cmap = cm.get_cmap('coolwarm')
    
    # create scatter plot with colors
    sc = ax.scatter(data_level.index, data_level[column], c=colors, cmap=cmap, alpha=0.8)
    
    # create color bar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.ax.set_ylabel('Latent Value')
    
    # text location 
    middle = (data_level[column].max()+ data_level[column].min()) / 2
    range = data_level[column].max() - data_level[column].min()
    text_level  = middle - range * 0.1

    for level in levels[levels != 0]:
        index_level = data_level[data_level['anomaly level'] == level].index[0]

        ax.axvline(x=index_level, color='steelblue', linestyle='--', alpha=0.5)
        ax.text(index_level, text_level, f'Anomaly level {level}', rotation=90, color='firebrick', alpha=0.9)
    ax.legend()
    ax.set_xlabel('Index')
    ax.set_ylabel(column)
    ax.set_title(f'Control Chart for System {system_id}')
    return ax