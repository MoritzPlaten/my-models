import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#from transformers import BertTokenizer
from Layers.Transformer import Transformer
from Dataset.MyDataset import generate_random_data

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
total_sequences = 64*2

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

input_seq, target_seq, input_padding_mask, target_padding_mask = generate_random_data(
    total_sequences, max_input_length, max_target_length, input_vocab_size, target_vocab_size, start_token_input=6, start_token_target=6
)

input_seq_len =  int(len(input_seq) - batch_size)
target_seq_len = int(len(target_seq) - batch_size)

X_train = input_seq[:input_seq_len]
y_train = target_seq[:target_seq_len]
X_validiation = input_seq[input_seq_len:]
y_validiation= target_seq[target_seq_len:]

history = transformer.my_train(X_train, y_train, X_validiation, y_validiation, epochs=2, batch_size=batch_size)

tf.saved_model.save(transformer, "transformer.keras", signatures={"my_predict": transformer.my_predict})
 
transformer.summary()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(history['loss'], label='Training Loss')
ax1.plot(history['val_loss'], label='Validation Loss')
ax1.set_title('Loss über Epochen')
ax1.set_xlabel('Epoche')
ax1.set_ylabel('Verlust')
ax1.legend()

ax2.plot(history['accuracy'], label='Training Accuracy')
ax2.plot(history['val_accuracy'], label='Validation Accuracy')
ax2.set_title('Genauigkeit über Epochen')
ax2.set_xlabel('Epoche')
ax2.set_ylabel('Genauigkeit')
ax2.legend()

plt.tight_layout()
plt.savefig("training_metrics_plot.png")
plt.show()