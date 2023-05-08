from PBSHM_mdof.data.formatting import format_data
from PBSHM_mdof.models.baseline import optht
import numpy as np 
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
from PBSHM_mdof.models.utils import compute_auc
from sklearn.decomposition import PCA

df, labels, index_train, index_test, transformer = format_data(test_size=0.5)
system_name = (df['system_name'].unique()).tolist()
system_name.sort(key=lambda x:int(x.split('_')[-1]))

df_pca = df[['system_name','anomaly_level','state','latent_value','resonance_freq']]
df_pca['type']= 'test'
df_pca.loc[index_train, 'type'] = 'train'
training_size = 700
noise= 0.02
remove_mode =8


def prepare_data_pca(df, system_name,noise):
    data_system = df[df['system_name'] == system_name]
    data_system.sort_values(by='type',ascending=False, inplace=True)
    data_system.sort_values(by='anomaly_level',ascending=True, inplace=True)
    data_system.reset_index(inplace=True)
    anomaly_level=data_system[data_system['anomaly_level'].astype(float).diff()!=0]['anomaly_level'].to_dict()
    all_res = np.stack(data_system['resonance_freq'].values)
    arg_sorted=all_res.argsort(axis=1)
    all_res=all_res[np.arange(all_res.shape[0])[:,None], arg_sorted]
    print(all_res.shape)
    all_res = all_res[:,:-1]
    print(all_res.shape)
    all_res = all_res + np.random.normal(0, noise, all_res.shape)
    training_res = all_res[0:training_size,:]

    std = np.std(training_res, axis=0)
    mean = np.mean(training_res, axis=0)
    all_res = (all_res - mean)/std
    training_res = (training_res - mean)/std
    return training_res, all_res, anomaly_level

def find_best_k(training_res):
    U, s, Vh = np.linalg.svd(training_res, full_matrices=False)
    k = optht.optht(training_res, sv=s, sigma=None)
    return k

def train_pca(training_res,all_res,k):
    pca = PCA(n_components=k)
    pca.fit(training_res)
    residual = all_res - pca.inverse_transform(pca.transform(all_res))
    residual = np.sum(np.square(residual), axis=1)
    return residual

def compute_auc_for_all_anomaly(anomaly_score,
                                training_size,
                                anomaly_level):
    end_of_normal_test = list(anomaly_level.keys())[1]
    normal_test = anomaly_score[training_size:end_of_normal_test]
    aucs = {}
    for i in range(0, len(anomaly_level)-1):
        start= list(anomaly_level.keys())[i]
        end = list(anomaly_level.keys())[i+1]
        unhealthy_test = anomaly_score[start:end]
        auc = compute_auc(unhealthy_test,normal_test)
        auc = round(auc, 3)
        aucs[anomaly_level[end]] = auc
    return aucs

with mlflow.start_run(experiment_id=0, run_name="pca_baseline"):
    training_res, all_res, anomaly_level=prepare_data_pca(df_pca, 'system_1',noise)
    aucs_df = pd.DataFrame(columns=system_name, index=list(anomaly_level.values())[1:])
    for system in system_name:
        training_res, all_res, anomaly_level=prepare_data_pca(df_pca, system,noise)
        k = find_best_k(training_res)
        anomaly_score = train_pca(training_res,all_res,k)
        aucs = compute_auc_for_all_anomaly(anomaly_score,training_size,anomaly_level)
        aucs_df[system] = aucs.values()
    aucs_interest=aucs_df.loc['0.03'].values
    mlflow.log_metric("aucs_0.03_mean", np.mean(aucs_interest))
    mlflow.log_metric("aucs_0.03_std", np.std(aucs_interest))
    mlflow.log_metric("aucs_0.03_min", np.min(aucs_interest))
    mlflow.log_metric("aucs_0.03_max", np.max(aucs_interest))
    mlflow.log_param("noise", noise)
    mlflow.log_param("training_size", training_size)
    mlflow.log_param("k", k)
    mlflow.log_param("remove_mode", remove_mode)
    aucs_df.to_csv('aucs_pca.csv')
    mlflow.log_artifact('aucs_pca.csv')

