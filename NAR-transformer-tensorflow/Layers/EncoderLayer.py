import keras

from Layers.GlobalSelfAttention import GlobalSelfAttention
from Layers.FeedForward import FeedForward

class EncoderLayer(keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, kernel_initializer, seed, dropout_rate=0.1):
    super().__init__()

    self.supports_masking = True

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate,
        seed=seed, kernel_initializer=kernel_initializer)

    self.ffn = FeedForward(d_model, dff, kernel_initializer=kernel_initializer, seed=seed)

  def call(self, x, training=False):
    x = self.self_attention(x, training=training)
    x = self.ffn(x)
    return x