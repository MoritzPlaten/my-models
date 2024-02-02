import tensorflow as tf
import keras

from .helper_class import Encoder, Decoder

class MyTransformer(keras.Model):

    def __init__(self, input_vocab_size, target_vocab_size, sequence_length=2048, num_layers=4, d_model=128, num_heads=8, dff=512, dropout=0.1):
        super(MyTransformer, self).__init__()

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

        self.mask = keras.layers.Masking(mask_value=0)

        self.encoder = Encoder(
            num_layers=num_layers,
            num_heads=num_heads,
            d_model=d_model,
            vocab_size=input_vocab_size,
            dff=dff,
            sequence_length=sequence_length,
            dropout=dropout
        )

        self.decoder = Decoder(
            num_layers=num_layers,
            num_heads=num_heads,
            d_model=d_model,
            vocab_size=target_vocab_size,
            dff=dff,
            sequence_length=sequence_length,
            dropout=dropout
        )

        self.attn_scores = None

        self.output_layer = keras.layers.Dense(target_vocab_size)

    def compute_loss(self, x, pred):

        mask = x != 0
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        loss = loss_object(x, pred)

        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask

        loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
        return loss

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


    def call(self, inputs):

        context, x = inputs

        #batch_size = tf.shape(x)[0]
        #sequence_length = tf.shape(x)[1]
        #mask = tf.linalg.band_part(tf.ones((batch_size, sequence_length, sequence_length)), -1, 0)
        #mask = mask * tf.cast(tf.math.not_equal(x, 0), tf.float32)
        #mask = tf.cast(tf.math.not_equal(x, 0), tf.float32)

        attention_mask = tf.cast(tf.math.not_equal(x, 0), tf.float32)
        attention_mask = tf.expand_dims(attention_mask, axis=1)
        mask = attention_mask * tf.transpose(attention_mask, perm=[0, 2, 1])

        x = self.mask(x)

        context = self.encoder(context, mask=mask)

        x = self.decoder(x, context, mask=mask)

        self.attn_scores = self.decoder.dec_layers[-1].last_attn_scores

        output = self.output_layer(x)

        try:
            del output._keras_mask
        except AttributeError:
            pass

        return output
