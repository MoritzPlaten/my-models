import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        self.supports_masking = True

        self.d_model = d_model
        self.max_seq_length = max_seq_length

        position = tf.range(max_seq_length, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / d_model))
        pe = tf.concat([tf.sin(position * div_term), tf.cos(position * div_term)], axis=-1)
        pe = tf.reshape(pe, [max_seq_length, d_model])
        self.pe = tf.Variable(pe, trainable=False, name='positional_encoding')

    def call(self, x):
        seq_length = tf.shape(x)[1]
        pe = self.pe[:, :seq_length]
        return x + pe


class PositionWiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, activation="relu"):
        super(PositionWiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = tf.keras.activations.get(activation)

        self.fc1 = tf.keras.layers.Dense(d_ff, activation=self.activation)
        self.fc2 = tf.keras.layers.Dense(d_model)

    def call(self, x):
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Input dimension {x.shape[-1]} doesn't match expected dimension {self.d_model}.")

        x = self.fc1(x)
        x = self.fc2(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        self.supports_masking = True

        self.multiHead = tf.keras.layers.MultiHeadAttention(
            key_dim=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        self.ff = PositionWiseFeedForward(d_model, d_ff=d_ff, activation=activation)
        self.add = tf.keras.layers.Add()
        self.norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, src, src_mask):
        att_output = self.multiHead([src, src, src])
        x = self.add([att_output, src])
        x = self.norm(x)
        x = self.dropout(x)

        ff_output = self.ff(x)
        x = self.add([ff_output, x])
        x = self.norm(x)
        x = self.dropout(x)
        return x


class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        self.supports_masking = True

        self.multiHead1 = tf.keras.layers.MultiHeadAttention(
            key_dim=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        self.add = tf.keras.layers.Add()
        self.norm = tf.keras.layers.LayerNormalization()

        self.multiHead2 = tf.keras.layers.MultiHeadAttention(
            key_dim=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        self.ff = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, activation=activation)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, tgt, enc_output, src_mask=None, tgt_mask=None):

        att_output1 = self.multiHead1([tgt, tgt, tgt])
        x = self.add([att_output1, tgt])
        x = self.norm(x)
        x = self.dropout(x)

        att_output2 = self.multiHead2([enc_output, enc_output, x])
        x = self.add([att_output2, x])
        x = self.norm(x)
        x = self.dropout(x)

        ff_output = self.ff(x)
        x = self.add([ff_output, x])
        x = self.norm(x)
        x = self.dropout(x)
        return x

