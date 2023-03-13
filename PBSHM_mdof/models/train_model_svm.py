#%% 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.svm import SVC
from PBSHM_mdof.data.formatting import format_data
from utils import plot_control_chart, compute_auc_df
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


# Load and format the data
df, labels, index_train, index_test, transformer = format_data(test_size=0.5)
psds_test = np.stack(df.loc[index_test]['psd'].values)
psds_train = np.stack(df.loc[index_train]['psd'].values)

system_id_test = labels[index_test]
system_id_train = labels[index_train]

num_class = system_id_train.shape[-1]
system_name = (df['system_name'].unique()).tolist()
system_name.sort(key=lambda x:x[-2:])

# Log variable values to MLflow
snr = 0
latent_std = np.around(df['latent_value'].std())
latent_mean = np.around(df['latent_value'].mean())
training_size = len(index_train/num_class)
heterogenity = 0
kernel = 'rbf'
C = 1
gamma = 'scale'
decision_function_shape='ovr'
class_weight = 'balanced'

# Start MLflow run
import mlflow

with mlflow.start_run():
    mlflow.log_param("SNR", snr)
    mlflow.log_param("Latent std", latent_std)
    mlflow.log_param("Training size", training_size)
    mlflow.log_param("Heterogenity", heterogenity)
    mlflow.log_param("Kernel", kernel)
    mlflow.log_param("C", C)
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("decision_function_shape", decision_function_shape)
    mlflow.log_param("class_weight", class_weight)

    # Train the model and log the training progress to MLflow
    logging.info("Training the model")
    svm_model = SVC(kernel=kernel, C=C, gamma=gamma, decision_function_shape=decision_function_shape, class_weight=class_weight)
    svm_model.fit(psds_train, np.argmax(system_id_train, axis=1))
    score = svm_model.decision_function(np.stack(df['psd'].values))
    predicted_class = np.argmax(score, axis=1)
    true_class = np.argmax(labels, axis=1)
    confusion = confusion_matrix(true_class, predicted_class)
    sns.heatmap(pd.DataFrame(confusion)).get_figure().savefig('docs/'+'confusion.png')
    mlflow.log_artifact('docs/'+'confusion.png', 'confusion_matrix')

    # Compute and log the AUCs
    logging.info("Computing the AUCs")
    confidence = score[np.arange(len(true_class)), true_class]
    auc_df = compute_auc_df(df, confidence, system_name, index_test)
    auc_df = auc_df.round(3)
    auc_df.to_html('auc.html')
    mlflow.log_artifact('auc.html','aucs')
    mlflow.log_metrics({'auc_0.05': auc_df.iloc[3,:].mean()})
    mlflow.log_metrics({'auc_0.07': auc_df.iloc[4,:].mean()})
    logging.info("Logging the F1 scores")
    # Compute and log the F1 scores
    f1_score_list = f1_score(true_class, predicted_class,average=None)
    for sys in system_name:
        mlflow.log_metrics({f'auc_{sys}': auc_df[sys]['0.05']})
        id = np.argmax(transformer['label_encder'].transform([[sys]]))
        mlflow.log_metrics({f'f1_score_{sys}': f1_score_list[id]})

    result = {}
    for id in system_name:
        index_system = df[df['system_name'] == id].index
        result[id] = {'confidence': confidence[index_system]}
        result[id]['anomaly level'] = df.loc[index_system]['anomaly_level'].astype(float).values
    
    for system_id in system_name:
        fig,ax =plot_control_chart(result,system_id)
        fig.savefig('docs/'+system_id+'.png')
        mlflow.log_artifact('docs/'+system_id+'.png','Control chart')
        plt.close(fig)
    # Generate and log control charts for each sys
# %%
