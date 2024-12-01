import keras
import tensorflow as tf
from tqdm import tqdm

from Layers.Encoder import Encoder
from Layers.Decoder import Decoder

from Metrics.TransformerMetrics import masked_accuracy, masked_loss

class Transformer(keras.Model):

  def __init__(self, *, num_layers, d_model, num_heads, dff,
               input_vocab_size, target_vocab_size, dropout_rate=0.1, learning_rate=0.001):
    super().__init__()

    self.supports_masking = True

    self.START_TOKEN = 6
    self.END_TOKEN = 120
    self.target_vocab_size = target_vocab_size
    self.input_vocab_size = input_vocab_size

    self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=input_vocab_size,
                           dropout_rate=dropout_rate)

    self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=target_vocab_size,
                           dropout_rate=dropout_rate)

    self.final_layer = keras.layers.Dense(target_vocab_size)

  def call(self, inputs):

    context, x  = inputs
    context = self.encoder(context)  # (batch_size, context_len, d_model)
    x = self.decoder(x, context)  # (batch_size, target_len, d_model)
    logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

    try:
      del logits._keras_mask
    except AttributeError:
      pass

    return logits
  
  
  @tf.function
  def train_step(self, context, target):

    with tf.GradientTape() as tape:
        logits = self.call((context, target))
        loss = masked_loss(target, logits)

    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    accuracy = masked_accuracy(target, logits)

    return loss, accuracy
  

  @tf.function
  def evaluate(self, context, target):

    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)

    start_token = tf.constant([self.START_TOKEN], dtype=tf.int64)
    output_array = output_array.write(0, start_token)

    for i in range(self.target_vocab_size): 
        
        output = tf.transpose(output_array.stack())
        predictions = self.decoder([context, output], training=False)
        predictions = predictions[-1, -1, :]

        predicted_id = tf.argmax(predictions, axis=-1)
        predicted_id = tf.squeeze(predicted_id)
        
        output_array = output_array.write(i + 1, predicted_id)
        
        if predicted_id == self.END_TOKEN:
            break

    final_output = output_array.stack()
    
    mask = tf.cast(target != 0, tf.float32)
    loss = masked_loss(target, final_output, mask)
    accuracy = masked_accuracy(target, final_output, mask)

    return loss, accuracy



  def my_train(self, x, y, val_x, val_y, batch_size=32, epochs=10):
      history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

      train_dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(buffer_size=1024).batch(batch_size)
      val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y)).batch(batch_size)

      for epoch in range(epochs):
          print(f"Epoch {epoch + 1}/{epochs}")
          
          train_loss_sum = 0.0
          train_accuracy_sum = 0.0
          num_train_batches = 0

          for context, target in tqdm(train_dataset, desc="Training"):
              loss, accuracy = self.train_step(context, target)

              train_loss_sum += loss
              train_accuracy_sum += accuracy
              num_train_batches += 1

          avg_train_loss = train_loss_sum / num_train_batches
          avg_train_accuracy = train_accuracy_sum / num_train_batches

          print(f"Training - Loss: {avg_train_loss.numpy():.4f}, Accuracy: {avg_train_accuracy.numpy():.4f}")

          val_loss_sum = 0.0
          val_accuracy_sum = 0.0
          num_val_batches = 0

          for val_context, val_target in tqdm(val_dataset, desc="Validation"):
              val_loss, val_accuracy = self.evaluate(val_context, val_target)

              val_loss_sum += val_loss
              val_accuracy_sum += val_accuracy
              num_val_batches += 1

          avg_val_loss = val_loss_sum / num_val_batches
          avg_val_accuracy = val_accuracy_sum / num_val_batches

          print(f"Validation - Loss: {avg_val_loss.numpy():.4f}, Accuracy: {avg_val_accuracy.numpy():.4f}")

          history["loss"].append(avg_train_loss.numpy())
          history["accuracy"].append(avg_train_accuracy.numpy())
          history["val_loss"].append(avg_val_loss.numpy())
          history["val_accuracy"].append(avg_val_accuracy.numpy())

      return history

  