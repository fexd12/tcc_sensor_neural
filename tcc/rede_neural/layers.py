import tensorflow as tf


class ExpandLayer(tf.keras.layers.Layer):

    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super(ExpandLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        ax = self.axis
        input_shape = list(input_shape)
        if ax < 0:
            ax = len(input_shape) + ax
        input_shape.insert(ax+1, 1)
        return tuple(input_shape)

    def call(self, inputs, **kwargs):
        return tf.keras.backend.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        return dict(axis=self.axis)
