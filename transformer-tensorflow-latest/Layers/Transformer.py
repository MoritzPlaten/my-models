import keras
import tensorflow as tf

from Layers.Encoder import Encoder
from Layers.Decoder import Decoder

from Metrics.TransformerMetrics import masked_accuracy, masked_loss

class Transformer(keras.Model):

  def __init__(self, *, num_layers, d_model, num_heads, dff,
               input_vocab_size, target_vocab_size, dropout_rate=0.1, learning_rate=0.001):
    super().__init__()

    self.supports_masking = True

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
    # To use a Keras model with `.fit` you must pass all your inputs in the
    # first argument.
    context, x  = inputs

    context = self.encoder(context)  # (batch_size, context_len, d_model)

    x = self.decoder(x, context)  # (batch_size, target_len, d_model)

    # Final linear layer output.
    logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      # b/250038731
      del logits._keras_mask
    except AttributeError:
      pass

    # Return the final output and the attention weights.
    return logits
  
  @tf.function
  def train_step(self, context, target):
    with tf.GradientTape() as tape:
        # Forward pass
        logits = self.call((context, target))

        # Compute the loss (use masked_loss)
        loss = masked_loss(target, logits)

    # Compute gradients and apply them
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    # Compute accuracy using masked accuracy
    accuracy = masked_accuracy(target, logits)

    return loss, accuracy

  def my_train(self, x, y, batch_size=32, epochs=10):
    # History dictionary to track loss and accuracy
    history = {"loss": [], "accuracy": []}

    # Create a dataset and batch it
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(buffer_size=1024).batch(batch_size)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Initialize epoch metrics
        loss_sum = 0.0
        accuracy_sum = 0.0
        num_batches = 0

        # Iterate over each batch
        for context, target in dataset:
            # Perform a training step
            loss, accuracy = self.train_step(context, target)

            # Track metrics
            loss_sum += loss
            accuracy_sum += accuracy
            num_batches += 1

        # Compute average metrics for the epoch
        avg_loss = loss_sum / num_batches
        avg_accuracy = accuracy_sum / num_batches

        print(f"Loss: {avg_loss.numpy():.4f}, Accuracy: {avg_accuracy.numpy():.4f}")

        # Record metrics in history
        history["loss"].append(avg_loss.numpy())
        history["accuracy"].append(avg_accuracy.numpy())

    return history
  