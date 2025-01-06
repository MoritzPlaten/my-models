import keras

from Layers.PositionalEmbedding import PositionalEmbedding
from Layers.EncoderLayer import EncoderLayer

class Encoder(keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads,
               dff, vocab_size, kernel_initializer, seed, dropout_rate=0.1):
    super().__init__()

    self.supports_masking = True

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(
        vocab_size=vocab_size, d_model=d_model)

    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate,
                     seed=seed, kernel_initializer=kernel_initializer)
        for _ in range(num_layers)]
    self.dropout = keras.layers.Dropout(dropout_rate, seed=seed)

  def call(self, x, training=False):
    # `x` is token-IDs shape: (batch, seq_len)
    x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

    # Add dropout.
    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training=training)

    return x  # Shape `(batch_size, seq_len, d_model)`.