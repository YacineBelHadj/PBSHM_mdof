from keras.layers import Input, Dense ,Dropout,BatchNormalization
from keras.models import Model
import tensorflow as tf
import tensorflow as tf
class DenseSignalClassifier(tf.keras.Model):
    def __init__(self, num_class:int,
                  inputDim:tuple, 
                  dense_layers:list[int], 
                  dropout_rate=0.2, 
                  batch_norm:bool=True,
                  activation='relu',
                  l1_reg=0.01,
                  temperature=1.0):  # Add temperature parameter
        
        super(DenseSignalClassifier, self).__init__()
        self.num_class=num_class
        self.inputDim=inputDim
        self.dense_layers=dense_layers
        self.dropout_rate=dropout_rate
        self.batch_norm = batch_norm
        self.l1_reg = l1_reg  # Store L1 regularization parameter
        self.temperature = temperature  # Store temperature parameter
        self.hidden_layers = []
        for unit in (dense_layers):
            self.hidden_layers.append(Dense(unit, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
            if self.batch_norm:
                self.hidden_layers.append(BatchNormalization())
            self.hidden_layers.append(Dropout(self.dropout_rate))
        self.final_layer = Dense(num_class, activation='softmax', kernel_regularizer=tf.keras.regularizers.l1(self.l1_reg))  # Add L1 regularization to the final layer
    
    # Rest of the class methods remain the same

    
    def call_encoder(self,input:tf.Tensor,training=False):
        h = input
        for layer in self.hidden_layers:
            h = layer(h)
        return h
    
    def call_classifier(self,input:tf.Tensor,training=False):
        logits = self.final_layer(input)
        # Apply temperature scaling to logits
        logits /= self.temperature
        probabilities = tf.nn.softmax(logits)
        return probabilities
    
    def call(self,input:tf.Tensor,training=False):
        h = self.call_encoder(input,training)
        out = self.call_classifier(h,training)
        return out

    def build_model(self, optimizer='adam', temperature=1.0, **kwargs):  # Add temperature parameter to build_model method
        self.temperature = temperature  # Set temperature parameter
        inputLayer = Input(shape=(self.inputDim))
        out = self(inputLayer)
        # Modify loss function to include temperature scaling
        
        model = Model(inputs=inputLayer, outputs=out)
        model.compile(optimizer=optimizer, loss=get_loss(self.temperature), metrics=['accuracy'], **kwargs)
        return model
    
    def get_config(self):
        config = {
            'num_class': self.num_class,
            'inputDim': self.inputDim,
            'dense_layers': self.dense_layers,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'temperature': self.temperature  # Include temperature parameter in config
        }
        base_config = super(DenseSignalClassifier, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
def get_loss(temperature=1.0):
    def loss_fn(y_true, y_pred):
            y_true = tf.cast(y_true, y_pred.dtype)
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1) / temperature
            return loss
    return loss_fn