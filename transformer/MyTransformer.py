import tensorflow as tf
import keras

from .helper_class import Encoder, Decoder

class MyTransformer(keras.Model):

    def __init__(self, input_vocab_size, target_vocab_size, num_layers=4, d_model=128, num_heads=8, dff=512, dropout=0.1):
        super(MyTransformer, self).__init__()

        self.encoder = Encoder(
            num_layers=num_layers,
            num_heads=num_heads,
            d_model=d_model,
            vocab_size=input_vocab_size,
            dff=dff,
            dropout=dropout
        )

        self.decoder = Decoder(
            num_layers=num_layers,
            num_heads=num_heads,
            d_model=d_model,
            vocab_size=target_vocab_size,
            dff=dff,
            dropout=dropout
        )

        self.attn_scores = None

        self.output_layer = keras.layers.Dense(target_vocab_size)

    def call(self, inputs):

        context, x = inputs

        context = self.encoder(context) # (batch_size, context_len, d_model)

        x = self.decoder(x, context) # (batch_size, target_len, target_vocab_size)

        self.attn_scores = self.decoder.dec_layers[-1].last_attn_scores

        output = self.output_layer(x)

        try:
            del output._keras_mask
        except AttributeError:
            pass

        return output
