import tensorflow as tf
import keras

from Dataset.MyDataset import generate_random_data
from Layers.Transformer import Transformer

# Parameters (adjust as needed)
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1
batch_size = 64
max_input_length = 50
max_target_length = 50
input_vocab_size = 30000
target_vocab_size = 30000
total_sequences = 1

transformer = tf.saved_model.load(export_dir="transformer.keras")

start_token = 6

input_seq, target_seq, input_padding_mask, target_padding_mask = generate_random_data(
    total_sequences, max_input_length, max_target_length, input_vocab_size, target_vocab_size, start_token_input=start_token, start_token_target=start_token
)

# Predict
prediction = transformer.signatures["predict"](x=input_seq)

print("Final predicted sequence:", prediction)
