import keras

from Layers.CausalSelfAttention import CausalSelfAttention
from Layers.CrossAttention import CrossAttention
from Layers.FeedForward import FeedForward

class DecoderLayer(keras.layers.Layer):
  def __init__(self,
               *,
               d_model,
               num_heads,
               dff, kernel_initializer, seed,
               dropout_rate=0.1):
    super(DecoderLayer, self).__init__()

    self.supports_masking = True

    self.causal_self_attention = CausalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate,
        seed=seed, kernel_initializer=kernel_initializer)

    self.cross_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate,
        seed=seed, kernel_initializer=kernel_initializer)

    self.ffn = FeedForward(d_model, dff, kernel_initializer=kernel_initializer, seed=seed)

  def call(self, x, context, training=False):
    
    x = self.causal_self_attention(x=x, training=training)
    x = self.cross_attention(x=x, context=context, training=training)

    # Cache the last attention scores for plotting later
    self.last_attn_scores = self.cross_attention.last_attn_scores

    x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
    return x