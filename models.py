from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils.conv_utils import conv_output_length
import matplotlib.pyplot as plt

# Spectral Method
class SpectralConvNet(tf.keras.Model):
    def __init__(self, input_shape = (520, 72, 1), **kwargs):
        super(SpectralConvNet, self).__init__()
        self.conv_layer_1 = tf.keras.layers.Conv2D(
                filters=6,
                kernel_size=(5, 5),
                input_shape=input_shape,
                padding='valid',
                activation=tf.nn.relu
                )
        self.pool_layer_1 = tf.keras.layers.MaxPooling2D(padding='same')
        self.conv_layer_2 = tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=(5, 5),
                padding='valid',
                activation=tf.nn.relu
                )
        self.pool_layer_2 = tf.keras.layers.MaxPooling2D(padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.fc_layer_1 = tf.keras.layers.Dense(
                units=120,
                activation=tf.nn.relu
                )
        self.fc_layer_2 = tf.keras.layers.Dense(
                units=84,
                activation=tf.nn.relu
                )
        # outputs the same as the input shape
        self.output_layer = tf.keras.layers.Dense(
                units=input_shape,
                activation=tf.nn.softmax
                )
        @tf.function
        def call(self, features):
            activation = self.conv_layer_1(features)
            activation = self.pool_layer_1(activation)
            activation = self.conv_layer_2(activation)
            activation = self.pool_layer_2(activation)
            activation = self.flatten(activation)
            activation = self.fc_layer_1(activation)
            activation = self.fc_layer_2(activation)
            output = self.output_layer(activation)
            return output
    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)


# Time Domain Method
class CausalAtrousConvolution1D(Conv1D):
    def __init__(self, filters, kernel_size, init='glorot_uniform', activation=None,
                 padding='valid', strides=1, dilation_rate=1, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, bias_constraint=None, use_bias=True, causal=False, **kwargs):
        super(CausalAtrousConvolution1D, self).__init__(filters,
                                                        kernel_size=kernel_size,
                                                        strides=strides,
                                                        padding=padding,
                                                        dilation_rate=dilation_rate,
                                                        activation=activation,
                                                        use_bias=use_bias,
                                                        kernel_initializer=init,
                                                        activity_regularizer=activity_regularizer,
                                                        bias_regularizer=bias_regularizer,
                                                        kernel_constraint=kernel_constraint,
                                                        bias_constraint=bias_constraint,
                                                        **kwargs)

        self.causal = causal
        if self.causal and padding != 'valid':
            raise ValueError("Causal mode dictates border_mode=valid.")

    def compute_output_shape(self, input_shape):
        input_length = input_shape[1]

        if self.causal:
            input_length += self.dilation_rate[0] * (self.kernel_size[0] - 1)

        length = conv_output_length(input_length,
                                    self.kernel_size[0],
                                    self.padding,
                                    self.strides[0],
                                    dilation=self.dilation_rate[0])

        return (input_shape[0], length, self.filters)

    def call(self, x):
        if self.causal:
            x = asymmetric_temporal_padding(x, self.dilation_rate[0] * (self.kernel_size[0] - 1), 0)
        return super(CausalAtrousConvolution1D, self).call(x)

