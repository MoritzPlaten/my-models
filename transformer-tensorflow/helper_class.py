import tensorflow as tf
import numpy as np
import keras

def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]
  depths = np.arange(depth)[np.newaxis, :]/depth

  angle_rates = 1 / (10000**depths)
  angle_rads = positions * angle_rates

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
  
  def __init__(self, vocab_size, d_model, sequence_length=2048):
    super(PositionalEmbedding, self).__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
    self.pos_encoding = positional_encoding(length=sequence_length, depth=d_model)
    self.supports_masking = True

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):

    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x


class CrossAttention(keras.layers.Layer):

    def __init__(self, key_dim, num_heads, dropout=0.1):
        super(CrossAttention, self).__init__()

        self.multi_head = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)
        self.norm = keras.layers.LayerNormalization()
        self.add = keras.layers.Add()
        self.supports_masking = True

    def call(self, x, context, mask=None):
       
       attn_output, attn_scores = self.multi_head(
          query=x,
          value=context,
          key=context,
          attention_mask=mask,
          return_attention_scores=True
        )
       
       self.last_attn_scores = attn_scores

       x = self.add([x, attn_output])
       x = self.norm(x)
       return x
        
class GlobalSelfAttention(keras.layers.Layer):
   
    def __init__(self, key_dim, num_heads, dropout=0.1):
        super(GlobalSelfAttention, self).__init__()

        self.multi_head = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)
        self.norm = keras.layers.LayerNormalization()
        self.add = keras.layers.Add()
        self.supports_masking = True

    def call(self, x, mask=None):
        
        attn_output = self.multi_head(
            query=x,
            value=x,
            key=x,
            attention_mask=mask
        )

        x = self.add([x, attn_output])
        x = self.norm(x)
        return x

class CausalSelfAttention(keras.layers.Layer):
   
    def __init__(self, key_dim, num_heads, dropout=0.1):
        super(CausalSelfAttention, self).__init__()

        self.multi_head = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)
        self.norm = keras.layers.LayerNormalization()
        self.add = keras.layers.Add()
        self.supports_masking = True

    def call(self, x, mask=None):

        #TODO: add Masking for the Multiheadattention layer in every other layers too

        attn_output = self.multi_head(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True,
            attention_mask=mask
        )
        
        x = self.add([x, attn_output])
        x = self.norm(x)
        return x
    
class FeedForward(keras.layers.Layer):
   
    def __init__(self, d_model, dff, dropout=0.1):
        super(FeedForward, self).__init__()

        self.feedForw = keras.models.Sequential([
           keras.layers.Dense(dff, activation="relu"),
           keras.layers.Dense(d_model),
           keras.layers.Dropout(dropout)
        ])
        self.feedForw.supports_masking = True

        self.norm = keras.layers.LayerNormalization()
        self.add = keras.layers.Add()
        self.supports_masking = True

    def call(self, x):
       
       ff_output = self.feedForw(x)
       x = self.add([x, ff_output])
       x = self.norm(x)
       return x

class EncoderLayer(keras.layers.Layer):
   
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
      super(EncoderLayer, self).__init__()

      self.global_attn = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout
        )
      
      self.ff = FeedForward(
        d_model=d_model,
        dff=dff,
        dropout=dropout
        )
      self.supports_masking = True
      
    def call(self, x, mask=None):
       
       x = self.global_attn(x, mask=mask)
       x = self.ff(x)
       return x
    
class Encoder(keras.layers.Layer):
   
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, sequence_length=2048, dropout=0.1):
      super(Encoder, self).__init__()

      self.num_layers = num_layers

      self.pos_embb = PositionalEmbedding(
         vocab_size=vocab_size,
         d_model=d_model,
         sequence_length=sequence_length
      )

      self.enc_layers = [
         EncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout=dropout
         )
         for _ in range(num_layers)
      ]

      self.drop = keras.layers.Dropout(dropout)
      self.supports_masking = True

    def call(self, x, mask=None):
       
        x = self.pos_embb(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask=mask)
        
        x = self.drop(x)
        return x
    
class DecoderLayer(keras.layers.Layer):
   
    def __init__(self, num_heads, d_model, dff, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_attn = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout
        )

        self.cross_attn = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout)

        self.ff = FeedForward(
            d_model=d_model,
            dff=dff,
            dropout=dropout
            )
        self.supports_masking = True
        
    def call(self, x, context, mask=None):

        x = self.causal_attn(x, mask=mask)
        x = self.cross_attn(x, context, mask=mask)

        self.last_attn_scores = self.cross_attn.last_attn_scores
       
        x = self.ff(x)
        return x
      
class Decoder(keras.layers.Layer):
   
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, sequence_length=2048, dropout=0.1):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.last_attn_scores = None


        self.pos_embb = PositionalEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            sequence_length=sequence_length
        )

        self.drop = tf.keras.layers.Dropout(dropout)

        self.dec_layers = [
            DecoderLayer(
            num_heads=num_heads,
            d_model=d_model,
            dff=dff,
            dropout=dropout
            )

            for _ in range(num_layers)
        ]
        self.supports_masking = True

    def call(self, x, context, mask=None):

        x = self.pos_embb(x)

        x = self.drop(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context, mask=mask)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        
        return x
