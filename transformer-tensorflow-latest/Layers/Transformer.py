import keras
import tensorflow as tf
from tqdm import tqdm
import numpy as np

from Layers.Encoder import Encoder
from Layers.Decoder import Decoder

from Metrics.TransformerMetrics import masked_accuracy, masked_loss, simple_loss, simple_accuracy

class Transformer(keras.Model):

  def __init__(self, *, num_layers, d_model, num_heads, dff,
               input_vocab_size, target_vocab_size, max_target_length, dropout_rate=0.1, learning_rate=0.001, start_token=6, end_token=120):
    super(Transformer, self).__init__()

    self.supports_masking = True

    self.num_layers = num_layers
    self.d_model = d_model
    self.num_heads = num_heads
    self.dff = dff
    self.input_vocab_size = input_vocab_size
    self.target_vocab_size = target_vocab_size
    self.max_target_length = max_target_length
    self.dropout_rate = dropout_rate
    self.learning_rate = learning_rate
    self.start_token = start_token
    self.end_token = end_token

    self.masked_loss_fn = masked_loss
    self.masked_accuracy_fn = masked_accuracy
    self.simple_loss_fn = simple_loss
    self.simple_accuracy_fn = simple_accuracy

    self.rand_initializer = "glorot_uniform"
    self.seed = 42

    self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=input_vocab_size,
                           dropout_rate=dropout_rate,
                           seed=self.seed, kernel_initializer=self.rand_initializer)

    self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=target_vocab_size,
                           dropout_rate=dropout_rate, seed=self.seed, kernel_initializer=self.rand_initializer)

    self.final_layer = keras.layers.Dense(target_vocab_size, kernel_initializer=self.rand_initializer)


  def call(self, inputs, training=False):

    context, x  = inputs
    context = self.encoder(context, training=training)  # (batch_size, context_len, d_model)
    x = self.decoder(x, context, training=training)  # (batch_size, target_len, d_model)
    logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

    try:
      del logits._keras_mask
    except AttributeError:
      pass

    return logits
  
  
  @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32), tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
  def train_step(self, context, target):

    with tf.GradientTape() as tape:
      logits = self.call((context, target), training=True) #TODO: Here is the issue with the seed_generator. Something is wrong if I want to save it. Normally it should be True
      loss = self.masked_loss_fn(target, logits)
      
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    accuracy = self.masked_accuracy_fn(target, logits)

    return loss, accuracy
  
  
  @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
  def predict(self, x):

    # Initialize output array
    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    output_array = output_array.write(0, self.start_token)
    
    # Define loop variables
    i = tf.constant(0, dtype=tf.int32)
    predicted_id = tf.constant(self.start_token, dtype=tf.int64)

    # Define the condition for the while loop: i < max_target_length
    def condition(i, predicted_id, output_array):
        return tf.logical_and(i < self.max_target_length - 1, tf.not_equal(predicted_id, self.end_token))

    # Define the body of the loop
    def body(i, predicted_id, output_array):
        output = tf.transpose(output_array.stack())
        output = tf.expand_dims(output, axis=0)

        predictions = self.call([x, output], training=False)
        predictions = predictions[-1, -1, :]

        predicted_id = tf.argmax(predictions, axis=-1)
        predicted_id = tf.squeeze(predicted_id)

        # Update the output array with the new token
        output_array = output_array.write(i + 1, predicted_id)

        # Increment the loop counter
        i = i + 1
        return i, predicted_id, output_array

    # Use tf.while_loop to iterate with the condition and body
    i, predicted_id, output_array = tf.while_loop(condition, body, [i, predicted_id, output_array])

    # Stack the final output array
    final_output = output_array.stack()
    return final_output

  @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32), tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
  def evaluate(self, context, target):
      # Create a map function to process each context in the batch
      def compute_loss_and_accuracy(input_seq, target_seq):
          input_seq = tf.expand_dims(input_seq, axis=0)  # Add batch dimension
          final_output = self.predict(input_seq)
          final_output = tf.squeeze(final_output)
          
          loss = self.simple_loss_fn(target_seq, final_output)
          accuracy = self.simple_accuracy_fn(target_seq, final_output)
          return loss, accuracy

      # Map the function over the entire batch (context)
      results = tf.map_fn(lambda x: compute_loss_and_accuracy(x[0], x[1]), (context, target), dtype=(tf.float32, tf.float32))

      # Extract losses and accuracies from the results
      losses, accuracies = results

      # Return the mean loss and accuracy
      return tf.reduce_mean(losses), tf.reduce_mean(accuracies)


  def train(self, x, y, val_x, val_y, batch_size=32, epochs=10):

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    train_dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(buffer_size=1024).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y)).batch(batch_size)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        train_loss_sum = 0.0
        train_accuracy_sum = 0.0
        num_train_batches = 0

        for context, target in tqdm(train_dataset, desc="Training"):
            loss, accuracy = self.train_step(context=context, target=target)

            train_loss_sum += loss
            train_accuracy_sum += accuracy
            num_train_batches += 1

        avg_train_loss = train_loss_sum / num_train_batches
        avg_train_accuracy = train_accuracy_sum / num_train_batches

        print(f"Training - Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_accuracy:.4f}")

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

        print(f"Validation - Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_accuracy:.4f}")

        history["loss"].append(avg_train_loss)
        history["accuracy"].append(avg_train_accuracy)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(avg_val_accuracy)

    return history

  