import tensorflow as tf

class CustomDropout(tf.keras.layers.Layer):
    def __init__(self, rate, seed=None, **kwargs):
        super(CustomDropout, self).__init__(**kwargs)
        self.rate = rate
        self.seed = seed

    def call(self, inputs, training=False):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate, seed=self.seed)
        return inputs 