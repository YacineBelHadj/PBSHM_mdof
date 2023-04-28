
#%%
import tensorflow as tf
from datetime import datetime
from PBSHM_mdof.data.formatting import format_data
from PBSHM_mdof.models.classification.dense_nn import DenseSignalClassifier, get_loss

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import mlflow
import mlflow.keras
import pandas as pd
import seaborn as sns
from utils import compute_auc, plot_control_chart, plot_control_chart_latent , compute_auc_df

#%%
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
loss_function = 'categorical_crossentropy'
layers =[512, 256, 128, 64, 32]
optimizer = 'adam'
dropout_rate = 0
model = "DenseSignalClassifier"
activation = 'ReLU'
batch_norm = True
l1_reg = 0.0001
# prepare dataframe for plotting
#%%
df_result = df[['system_name','anomaly_level','state','latent_value']]
df_result['type']= 'test'
df_result.loc[index_train, 'type'] = 'train'

# Start MLflow run

#%%
with mlflow.start_run():
    mlflow.log_param("SNR", snr)
    mlflow.log_param("Latent std", latent_std)
    mlflow.log_param("Training size", training_size)
    mlflow.log_param("Heterogenity", heterogenity)
    mlflow.log_param("Loss function", loss_function)
    mlflow.log_param("Layers", layers)
    mlflow.log_param("depth", len(layers))
    mlflow.log_param("model_name", model)
    mlflow.log_param("optimizer", optimizer)
    mlflow.log_param("dropout_rate", dropout_rate)
    mlflow.log_param("batch_norm", batch_norm)
    mlflow.log_param("training_size", training_size)
    mlflow.log_param("latent_mean", latent_mean)
    mlflow.log_param("activation", activation)
    mlflow.log_param("l1_reg", l1_reg)

    # Define the model and log it to MLflow
    model_ = eval(model)(inputDim=(psds_train[0].shape[-1]),
                                    num_class=system_id_train.shape[-1],
                                    dense_layers=layers,
                                    dropout_rate=dropout_rate,
                                    batch_norm=batch_norm,
                                    activation=activation,
                                    l1_reg=l1_reg
                                    )
    model_=model_.build_model(optimizer=optimizer)

    # Log the model architecture to MLflow
    mlflow.keras.log_model(model_, "model")

    # Define callbacks
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="best_model.h5",
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        restore_best_weights=True)
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                     factor=0.5, 
                                                     patience=5, 
                                                     min_lr=1e-6)

    # Train the model and log the training progress to MLflow
    mlflow.keras.autolog()
    metrics={}
    model_.fit(psds_train, system_id_train, epochs=100, batch_size=64,verbose=1,validation_split=0.2,
              callbacks=[early_stopping_callback,reduce_lr,model_checkpoint_callback])
    best_model = tf.keras.models.load_model('best_model.h5',
                                             custom_objects={'DenseSignalClassifier': DenseSignalClassifier, 
                                                             'loss_fn': get_loss(temperature=1)})


    # Log the trained model to MLflow
    loss, accuracy = best_model.evaluate(psds_test, system_id_test, verbose=1)
    metrics['accuracy'] = accuracy
    mlflow.log_metrics(metrics)

    score = best_model.predict(np.stack(df['psd'].values))
    true_class = np.argmax(labels, axis=1)
    predicted_class = np.argmax(score, axis=1)
    confusion = confusion_matrix(true_class[index_test], predicted_class[index_test])
    sns.heatmap(pd.DataFrame(confusion)).get_figure().savefig('docs/'+'confusion.png')
    mlflow.log_artifact('docs/'+'confusion.png', 'confusion_matrix')
    #%%
    confidence=score[np.arange(len(true_class)), true_class]
    df_result['confidence'] = confidence

    auc_df =compute_auc_df(df_result)
    auc_df.to_html('auc.html')
    mlflow.log_artifact('auc.html','aucs')
    mlflow.log_metrics({'auc_0.05': auc_df.iloc[3,:].mean()})
    mlflow.log_metrics({'auc_0.07': auc_df.iloc[4,:].mean()})
    f1_score_list = f1_score(true_class, predicted_class,average=None)
    for sys in system_name:
        mlflow.log_metrics({f'auc_{sys}': auc_df[sys]['auc_0.05']})
        id = np.argmax(transformer['label_encder'].transform([[sys]]))
        mlflow.log_metrics({f'f1_score_{sys}': f1_score_list[id]})

    result = {}
    for id in system_name:
        index_system = df[df['system_name'] == id].index
        result[id] = {'confidence': confidence[index_system]}
        result[id]['anomaly level'] = df.loc[index_system]['anomaly_level'].astype(float).values
    
    for system_id in system_name:
        fig,ax =plot_control_chart(df_result,system_id)
        fig.savefig('docs/'+system_id+'.png')
        mlflow.log_artifact('docs/'+system_id+'.png','Control chart')
        plt.close(fig)
# %%
    df_resfreq = pd.DataFrame(np.stack(df['resonance_freq'].values))



    result_freq = {}
    for id in system_name:
        index_system = df[df['system_name'] == id].index
        result_freq[id] = {f'freq_{i}': freq for i,freq in enumerate(np.stack(df_resfreq.loc[index_system].values).T)}
        result_freq[id]['anomaly level'] = df.loc[index_system]['anomaly_level'].astype(float).values
        result_freq[id]['system_name'] = df.loc[index_system]['system_name'].values
        result_freq[id]['latent_value'] = df.loc[index_system]['latent_value'].values
    # %%
    plot_control_chart_latent('system_14', result_freq,column='freq_7',ylabel='Resonance frequency of mode 7(Hz)')
    plt.show()
    plt.close()




# %%
# %load_ext autoreload
# %autoreload 2
# %%
