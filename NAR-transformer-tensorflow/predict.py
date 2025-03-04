import tensorflow as tf

from Dataset.MyDataset import generate_random_data

batch_size = 64
max_input_length = 50
max_target_length = 50
input_vocab_size = 30000
target_vocab_size = 30000
total_sequences = 1

transformer = tf.saved_model.load('transformer.keras')

start_token = 6

input_seq, target_seq, input_padding_mask, target_padding_mask = generate_random_data(
    total_sequences, max_input_length, max_target_length, input_vocab_size, target_vocab_size
)

prediction = transformer.signatures["predict"](context=input_seq)

print("Final predicted sequence:", prediction)
