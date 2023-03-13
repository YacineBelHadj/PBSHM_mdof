from keras.layers import Input, Dense ,Dropout,BatchNormalization
from keras.models import Model
import tensorflow as tf



class DenseSignalClassifier(tf.keras.Model):
    def __init__(self, num_class:int,
                  inputDim:tuple, 
                  dense_layers:list[int], 
                  dropout_rate=0.2, 
                  batch_norm:bool=True,
                  activation='relu'):
        
        super(DenseSignalClassifier, self).__init__()
        self.num_class=num_class
        self.inputDim=inputDim
        self.dense_layers=dense_layers
        self.dropout_rate=dropout_rate
        self.batch_norm = batch_norm
        self.hidden_layers = []
        for unit in (dense_layers):
            self.hidden_layers.append(Dense(unit, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
            if self.batch_norm:
                self.hidden_layers.append(BatchNormalization())
            self.hidden_layers.append(Dropout(self.dropout_rate))
        self.final_layer = Dense(num_class, activation='softmax')

    def call(self,input:tf.Tensor,training=False):
        h = input
        for layer in self.hidden_layers:
            h = layer(h)
        out = self.final_layer(h)
        return out

    def encoder(self,input:tf.Tensor,training=False):
        h = input
        for layer in self.hidden_layers:
            h = layer(h)
        return h

    def build_model(self, optimizer='adam', loss='categorical_crossentropy', **kwargs):
        inputLayer = Input(shape=(self.inputDim))
        out = self(inputLayer)
        model = Model(inputs=inputLayer, outputs=out)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'], **kwargs)
        return model
    def get_config(self):
        config = {
            'num_class': self.num_class,
            'inputDim': self.inputDim,
            'dense_layers': self.dense_layers,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm
        }
        base_config = super(DenseSignalClassifier, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
