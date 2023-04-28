from sklearn.metrics import roc_auc_score
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.pyplot as plt

params = {'axes.labelsize': 16,
          'axes.titlesize': 14,
          'xtick.labelsize': 14,
          'ytick.labelsize': 17,
          'axes.linewidth': 1.5,
          'grid.linewidth': 1.2,
          'legend.fontsize': 16,
          'savefig.dpi': 100,
          'font.size': 16}

plt.rcParams.update(params)



def compute_auc_df(df,col_name='confidence'):
    system_name = df['system_name'].unique().tolist()
    system_name.sort(key=lambda x: int(x.split('_')[-1]))
    anomaly_level = np.sort(df['anomaly_level'].unique())
    auc_df = pd.DataFrame(columns=system_name)
    for system_id in system_name:
        healthy_training_data = df[col_name][(df['system_name'] == system_id) & (df['type'] == 'train')]
        healthy_testing_data = df[col_name][(df['system_name'] == system_id) & (df['type'] == 'validation')]
        for al in anomaly_level:
            if al == 0:
                auc = compute_auc(healthy_training_data, healthy_testing_data)
            else:
                healthy_testing_data = df[col_name][(df['system_name'] == system_id) & (df['type'] == 'validation')]
                anomaly_data = df[col_name][(df['system_name'] == system_id) & (df['type'] == 'test') & (df['anomaly_level'] == al)]
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

def plot_control_chart(statstic: np.ndarray, anomaly_level: dict, training_lim:int=700, ax=None, title='Q statistic'):
    """
    Plot the control chart of the given statistic
    Args:
        statstic: the statistic to plot
        anomaly_level: the anomaly level to plot
        ax: the axis to plot
        title: the title of the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 5))
    ucl = np.mean(statstic[:training_lim]) + 3 * np.std(statstic[:training_lim])
    ax.plot(statstic,marker='o',linestyle='', markersize=3)

    text_loc = get_text_location(ax)

    for k, v in list(anomaly_level.items())[1:]:
        ax.axvline(x=k, color='firebrick', linestyle='--', alpha=0.5)
        ax.text(k, text_loc, v, color='black', alpha=1, rotation=90)
    ax.axhline(y=ucl, color='firebrick', linestyle='--', alpha=0.5)
    ax.axvline(x=training_lim, color='black', linestyle='--', alpha=0.5)
    ax.text (training_lim, text_loc, 'training', color='black', alpha=1, rotation=90)
    if title is not None:
        ax.set_title(title)
    allert = np.where(statstic > ucl)[0]
    if len(allert) > 0:
        ax.plot(allert, statstic[allert], marker='o',linestyle='', color='firebrick', markersize=3)
    return ax
def get_text_location(ax,shift:float=0.1):
    y_lowbound, y_upbound= ax.get_ylim()
    middle = (y_lowbound + y_upbound) / 2
    range = (y_upbound - y_lowbound) / 2
    text_y = middle - range * shift
    return text_y

def plot_control_chart(df,system_id, column='confidence', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    data_system = df[df['system_name'] == system_id]
    df_train = data_system[data_system['type'] == 'train']
    df_test = data_system[data_system['type'] == 'test']
    df_test= df_test.sort_values(by='anomaly_level', ascending=True)

    df_plot = pd.concat([df_train, df_test])
    df_plot.reset_index(inplace=True)
    #df_plot[column]=df_plot[column].rolling(10).apply(lambda x:np.prod(x))
    ax.plot(df_plot.index, df_plot[column], marker='.', linestyle='')
    anomaly_level=df_plot[df_plot['anomaly_level'].astype(float).diff()!=0]['anomaly_level'].to_dict()

    y_lowbound, y_upbound= ax.get_ylim()
    middle = (y_lowbound + y_upbound) / 2
    range = (y_upbound - y_lowbound) / 2
    text_y = middle - range * 0.1


    ax.axvline(x=len(df_train), color='firebrick', linestyle='--', alpha=0.5)
    ax.text(len(df_train), text_y, 'test data', rotation=90, color='firebrick', alpha=0.9)
    for key in list(anomaly_level.keys())[1:]:
        ax.axvline(x=key, color='firebrick', linestyle='--', alpha=0.5)
        ax.text(key, text_y, anomaly_level[key], color='black', alpha=1,rotation=90)

    mean = np.mean(df_plot.iloc[:len(df_train)][column])
    std = np.std(df_plot.iloc[:len(df_train)][column])
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

def plot_control_chart_latent(system_id, result, column='freq_5',ylabel=None, ax=None):
    if ylabel is None:
        ylabel = column
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        
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
    middle = (data_level[column].max() + data_level[column].min()) / 2
    range = data_level[column].max() - data_level[column].min()
    text_level = middle - range * 0.1
    
    anomaly_level = data_level[data_level['anomaly level'].astype(float).diff() != 0]['anomaly level'].to_dict()
    for key in list(anomaly_level.keys())[1:]:
        ax.axvline(x=key, color='firebrick', linestyle='--', alpha=0.5)
        ax.text(key, text_level, anomaly_level[key], rotation=90, color='black', alpha=1)

    ax.set_xlabel('Index')
    ax.set_ylabel(ylabel)
    ax.set_title(f'Control Chart for System {system_id}')
    return ax