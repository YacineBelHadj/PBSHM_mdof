from keras.layers import Input, Dense , Conv1D, Flatten,MaxPooling1D,GlobalAveragePooling1D
from keras.models import Model
import tensorflow as tf



class ConvSignalClassifier(Model):
    def __init__(self, inputDim, num_class, conv_layers, dense_layers, **kwargs):
        super(ConvSignalClassifier, self).__init__(**kwargs)
        self.inputDim = inputDim
        self.num_class = num_class
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers

        # Define the convolutional layers
        self.conv_blocks = []
        for filters, kernel_size in conv_layers:
            self.conv_blocks.append(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))

        # Define the pooling layers
        self.pool_blocks = []
        for _ in conv_layers:
            self.pool_blocks.append(MaxPooling1D(pool_size=2))

        # Define the global average pooling layer
        self.global_avg_pool = GlobalAveragePooling1D()

        # Define the dense layers
        self.dense_blocks = []
        for units in dense_layers:
            self.dense_blocks.append(Dense(units=units, activation='relu'))

        # Define the final output layer
        self.output_layer = Dense(units=num_class, activation='softmax')

    def call(self, inputs, training=False):
        # Apply the convolutional layers
        x = inputs
        for conv_block, pool_block in zip(self.conv_blocks, self.pool_blocks):
            x = conv_block(x)
            x = pool_block(x)

        # Apply the global average pooling layer
        x = self.global_avg_pool(x)

        # Apply the dense layers
        for dense_block in self.dense_blocks:
            x = dense_block(x)

        # Apply the output layer
        x = self.output_layer(x)

        return x

    def build_model(self, lr=0.001, **kwargs):
        input_layer = Input(shape=self.inputDim)
        output_layer = self.call(input_layer)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'], **kwargs)
        return model