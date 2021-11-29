import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras.losses import Reduction


class ReverseHuber(Loss):

    def __init__(self, slope, delta, name="reverse_huber", reduction=Reduction.AUTO):
        super().__init__()
        self.slope = slope
        self.delta = delta
        self.name = name
        self.reduction = reduction

    def call(self, y_true, y_pred):
        y_pred = y_pred[:, :, :, 0]
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        y_true = tf.cast(y_true, dtype=tf.float32)
        delta = tf.cast(self.delta, dtype=tf.float32)
        slope = tf.cast(self.slope, dtype=tf.float32)
        error = tf.subtract(y_pred, y_true)
        abs_error = tf.abs(error)
        return tf.math.reduce_mean(
            tf.where(abs_error <= delta, slope * abs_error,
                     slope * tf.square(abs_error) / (2 * delta) + slope * delta / 2),
            axis=-1)
