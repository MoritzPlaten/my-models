import keras
import tensorflow as tf

def masked_loss(label, pred):
  mask = label != 0
  loss_object = keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss


def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)
  match = label == pred

  mask = label != 0

  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)



def simple_loss(y_true, y_pred):
    
    y_true = tf.cast(y_true, dtype=tf.int32)
    y_pred = tf.cast(y_pred, dtype=tf.int32)
    
    loss = tf.cast(tf.not_equal(y_true, y_pred), dtype=tf.float32)
    return tf.reduce_mean(loss)



def simple_accuracy(y_true, y_pred):
    
    assert y_true.shape == y_pred.shape
    
    accuracy = tf.cast(tf.equal(y_true, y_pred), dtype=tf.float32)
    accuracy = tf.reduce_mean(accuracy)
    return accuracy