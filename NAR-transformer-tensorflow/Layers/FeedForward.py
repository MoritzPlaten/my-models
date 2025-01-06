import keras

class FeedForward(keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1, seed=42):
    super().__init__()

    self.supports_masking = True

    self.seq = keras.Sequential([
      keras.layers.Dense(dff, activation='relu'),
      keras.layers.Dense(d_model),
      keras.layers.Dropout(dropout_rate, seed=seed)
    ])
    self.seq.supports_masking = True

    self.add = keras.layers.Add()
    self.add.supports_masking = True

    self.layer_norm = keras.layers.LayerNormalization()
    self.layer_norm.supports_masking = True

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x) 
    return x