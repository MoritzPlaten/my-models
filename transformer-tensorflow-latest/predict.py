import tensorflow as tf
import numpy as np


from Layers.Transformer import Transformer
from Metrics.TransformerMetrics import masked_accuracy, masked_loss  # Import your custom metrics
from Dataset.MyDataset import generate_random_data

# Parameters (adjust as needed)
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1
batch_size = 64  # Set to 1 for inference (predicting one example at a time)
max_input_length = 50
max_target_length = 50
input_vocab_size = 30000
target_vocab_size = 30000
total_sequences = 1

# Instantiate the transformer model
transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=input_vocab_size,
    target_vocab_size=target_vocab_size,
    dropout_rate=dropout_rate,
    max_target_length=max_target_length
)

transformer.load_weights("transformer.weights.h5")

start_token = 6

input_seq, target_seq, input_padding_mask, target_padding_mask = generate_random_data(
    total_sequences, max_input_length, max_target_length, input_vocab_size, target_vocab_size
)

prediction = transformer.my_predict(input_seq)

print("Final predicted sequence:", prediction.numpy())
