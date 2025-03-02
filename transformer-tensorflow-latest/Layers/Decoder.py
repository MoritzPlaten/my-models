import keras

from Layers.PositionalEmbedding import PositionalEmbedding
from Layers.DecoderLayer import DecoderLayer
from Layers.CustomDropout import CustomDropout

class Decoder(keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, kernel_initializer, seed,
               dropout_rate=0.1):
    super(Decoder, self).__init__()

    self.supports_masking = True

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                             d_model=d_model)
    self.dropout = CustomDropout(dropout_rate, seed=seed)
    self.dec_layers = [
        DecoderLayer(d_model=d_model, num_heads=num_heads,
                     dff=dff, dropout_rate=dropout_rate, seed=seed, kernel_initializer=kernel_initializer)
        for _ in range(num_layers)]

    self.last_attn_scores = None

  def call(self, x, context, training=False):
    # `x` is token-IDs shape (batch, target_seq_len)
    x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x  = self.dec_layers[i](x, context, training=training)

    self.last_attn_scores = self.dec_layers[-1].last_attn_scores

    # The shape of x is (batch_size, target_seq_len, d_model).
    return x