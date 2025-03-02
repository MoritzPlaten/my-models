import keras

from Layers.MultiheadAttention import CustomMultiHeadAttention

class BaseAttention(keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()

    self.supports_masking = True

    #self.mha = keras.layers.MultiHeadAttention(**kwargs)
    self.mha = CustomMultiHeadAttention(**kwargs)
    self.layernorm =keras.layers.LayerNormalization()
    self.add = keras.layers.Add()