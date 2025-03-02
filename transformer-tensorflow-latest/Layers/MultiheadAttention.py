import tensorflow as tf

class CustomMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, value_dim=None, dropout=0.0, seed=None, kernel_initializer="glorot_uniform", **kwargs):
        """
        Custom MultiHeadAttention layer for Transformer models.

        Args:
            num_heads (int): Number of attention heads.
            key_dim (int): Dimensionality of the query and key vectors.
            value_dim (int, optional): Dimensionality of the value vectors. Defaults to key_dim.
            dropout (float, optional): Dropout rate to apply to attention scores. Defaults to 0.0.
            seed (int, optional): Random seed for dropout. Defaults to None.
            kernel_initializer (str or tf.keras.initializers.Initializer, optional): Initializer for kernel weights. Defaults to "glorot_uniform".
        """
        super(CustomMultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim if value_dim is not None else key_dim
        self.dropout = dropout
        self.seed = seed
        self.kernel_initializer = kernel_initializer

        # Define layers with kernel initializer
        self.query_dense = tf.keras.layers.Dense(
            num_heads * key_dim, kernel_initializer=kernel_initializer
        )
        self.key_dense = tf.keras.layers.Dense(
            num_heads * key_dim, kernel_initializer=kernel_initializer
        )
        self.value_dense = tf.keras.layers.Dense(
            num_heads * self.value_dim, kernel_initializer=kernel_initializer
        )
        self.output_dense = tf.keras.layers.Dense(
            num_heads * self.value_dim, kernel_initializer=kernel_initializer
        )

        # Dropout with seed
        self.attention_dropout = tf.keras.layers.Dropout(dropout, seed=seed)

    def build(self, input_shape):
        """
        Create weights for the layer.
        """
        super(CustomMultiHeadAttention, self).build(input_shape)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result to shape (batch_size, num_heads, seq_len, depth).
        """
        depth = x.shape[-1] // self.num_heads
        x = tf.reshape(x, (batch_size, -1, self.num_heads, depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def combine_heads(self, x, batch_size):
        """
        Combine heads by transposing and reshaping back to (batch_size, seq_len, num_heads * depth).
        """
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (batch_size, -1, self.num_heads * x.shape[-1]))

    def scaled_dot_product_attention(self, query, key, value, mask):
        """
        Compute scaled dot-product attention.
        """
        matmul_qk = tf.matmul(query, key, transpose_b=True)  # [batch_size, num_heads, seq_len_q, seq_len_k]
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(depth)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)  # Mask out invalid positions

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # [batch_size, num_heads, seq_len_q, seq_len_k]
        attention_weights = self.attention_dropout(attention_weights)

        output = tf.matmul(attention_weights, value)  # [batch_size, num_heads, seq_len_q, depth]
        return output, attention_weights

    def call(self, query, key, value, mask=None, training=None):
        """
        Forward pass for the MultiHeadAttention layer.

        Args:
            inputs: A tuple of (query, key, value).
            mask: A mask tensor to apply during attention computation.
            training: Whether the layer is in training mode.
        """
        batch_size = tf.shape(query)[0]

        # Dense layers for query, key, and value
        query = self.query_dense(query)  # [batch_size, seq_len, num_heads * key_dim]
        key = self.key_dense(key)
        value = self.value_dense(value)

        # Split heads
        query = self.split_heads(query, batch_size)  # [batch_size, num_heads, seq_len, key_dim]
        key = self.split_heads(key, batch_size)      # [batch_size, num_heads, seq_len, key_dim]
        value = self.split_heads(value, batch_size)  # [batch_size, num_heads, seq_len, value_dim]

        # Scaled dot-product attention
        attention_output, attention_weights = self.scaled_dot_product_attention(query, key, value, mask)

        # Combine heads
        attention_output = self.combine_heads(attention_output, batch_size)  # [batch_size, seq_len, num_heads * value_dim]

        # Final dense layer
        output = self.output_dense(attention_output)  # [batch_size, seq_len, d_model]
        return output, attention_weights

    def get_config(self):
        """
        Serialize the layer's configuration for saving.
        """
        config = super(CustomMultiHeadAttention, self).get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "dropout": self.dropout,
            "seed": self.seed,
            "kernel_initializer": self.kernel_initializer,
        })
        return config
