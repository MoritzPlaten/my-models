import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#from transformers import BertTokenizer
from Layers.Transformer import Transformer
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
total_sequences = 500000

# Instantiate the transformer model
transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=input_vocab_size,
    target_vocab_size=target_vocab_size,
    dropout_rate=dropout_rate
)

input_seq, target_seq, input_padding_mask, target_padding_mask = generate_random_data(
    total_sequences, max_input_length, max_target_length, input_vocab_size, target_vocab_size
)

input_seq_len =  int(len(input_seq) * 0.9)
target_seq_len = int(len(target_seq) * 0.9)

X_train = input_seq[:input_seq_len]
y_train = target_seq[:target_seq_len]
X_validiation = input_seq[input_seq_len:]
y_validiation= target_seq[target_seq_len:]

history = transformer.my_train(X_train, y_train, X_validiation, y_validiation, epochs=2)

# Verlust plotten
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Loss über Epochen')
plt.xlabel('Epoche')
plt.ylabel('Verlust')
plt.legend()
plt.show()

# Genauigkeit plotten
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Genauigkeit über Epochen')
plt.xlabel('Epoche')
plt.ylabel('Genauigkeit')
plt.legend()
plt.show()

# After training, you can save the model if needed
transformer.save_weights('transformer.weights.h5')

# Print model summary
transformer.summary()