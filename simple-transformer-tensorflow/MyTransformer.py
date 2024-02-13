import tensorflow as tf

from helperClass import PositionalEncoding, EncoderLayer, DecoderLayer


class MyTransformer(tf.keras.Model):

    def __init__(self, src_vocab_size=5000, tgt_vocab_size=5000, d_model=512, num_heads=8,
                 num_layers=6, d_ff=2048, max_seq_length=100, dropout=0.1, activation="relu"):
        super(MyTransformer, self).__init__()
        self.supports_masking = True
        self.num_layers = num_layers
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

        self.masking = tf.keras.layers.Masking(mask_value=0.0)

        self.emb_src = tf.keras.layers.Embedding(src_vocab_size, d_model)
        self.emb_tgt = tf.keras.layers.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_seq_length=max_seq_length)

        self.enc_layers = [
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation
            ) for _ in range(num_layers)
        ]

        self.dec_layers = [
            DecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation
            ) for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(dropout)

        self.output_layer = tf.keras.layers.Dense(tgt_vocab_size)

    def generate_mask(self, src, tgt):

        padding_mask_src = tf.math.logical_not(tf.equal(src, 0))
        padding_mask_tgt = tf.math.logical_not(tf.equal(tgt, 0))

        seq_length = tf.shape(tgt)[-1]

        nopeak_mask = tf.linalg.band_part(tf.ones((1, seq_length, seq_length)), -1, 0)
        padding_mask_tgt = tf.logical_and(padding_mask_tgt, nopeak_mask)

        src_mask = tf.expand_dims(tf.expand_dims(padding_mask_src, 1), 1)
        tgt_mask = tf.expand_dims(padding_mask_tgt, 1)

        return src_mask, tgt_mask

    def train_step(self, data):
        context, x = data

        with tf.GradientTape() as tape:
            y_pred = self([context, x], training=True)

            loss = self.compute_loss(context, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.acc_metric.update_state(context, y_pred)
        return {"loss": self.loss_tracker.result(), "acc": self.acc_metric.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.acc_metric]

    def compute_loss(self, x, pred):

        mask = x != 0
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        loss = loss_object(x, pred)

        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask

        loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
        return loss

    def call(self, inputs):

        src, tgt, = inputs

        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src = self.masking(src)
        src = self.emb_src(src)
        src = self.pos_encoding(src)
        src = self.dropout(src)

        tgt = self.masking(tgt)
        tgt = self.emb_src(tgt)
        tgt = self.pos_encoding(tgt)
        tgt = self.dropout(tgt)

        enc_output = src
        for enc_layer in self.enc_layers:
            enc_output = enc_layer(src, src_mask=src_mask)

        dec_output = tgt
        for dec_layer in self.dec_layers:
            dec_output = dec_layer(tgt=tgt, enc_output=enc_output, src_mask=src_mask, tgt_mask=tgt_mask)

        output = self.output_layer(dec_output)
        return output
