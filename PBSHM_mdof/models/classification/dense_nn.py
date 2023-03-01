from keras.layers import Input, Dense 
from keras.models import Model
import tensorflow as tf

def compose(*funcs):
    def inner(x):
        for f in funcs:
            x = f(x)
        return x
    return inner

class DenseSignalClassifier(tf.keras.Model):
    def __init__(self,num_class:int,inputDim:tuple,dense_layers:list[int]):
        super(DenseSignalClassifier, self).__init__()
        self.num_class=num_class
        self.inputDim=inputDim
        self.dense_layers=dense_layers
        self.hidden_layers = []
        for unit in (dense_layers):
            self.hidden_layers.append(Dense(unit,activation='relu'))
        self.final_layer = Dense(num_class,activation='softmax')

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

    def build_model(self,loss='categorical_crossentropy',**kwargs):
        inputLayer = Input(shape=(self.inputDim))
        out = self(inputLayer)
        model = Model(inputs=inputLayer, outputs=out)
        model.compile(optimizer='adam', loss=, metrics=['accuracy'],**kwargs)
        return model

if __name__=='__main__':
    model = DenseSignalClassifier(inputDim=(1500,), num_class=12,dense_layers=[2048,1024,512,256,128,64,32])
    model.build_model()