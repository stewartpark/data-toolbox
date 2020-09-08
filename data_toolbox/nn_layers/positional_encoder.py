from tensorflow.keras.layers import Layer, Add
import tensorflow as tf
import numpy as np


class PositionalEncoder(Layer):

    def __init__(self, max_length, dtype='float32'):
        super(PositionalEncoder, self).__init__()
        self.add = Add()
        self._max_length = max_length
        self._dtype = dtype
        self._positional_matrix = None

    def calculate_positional_matrix(self, embedding_size):
        if self._positional_matrix is None:
            pe = np.zeros((self._max_length, embedding_size),
                          dtype=self._dtype)
            for pos in range(self._max_length):
                for i in range(embedding_size):
                    if i % 2 == 0:
                        pe[pos, i] = np.sin(
                            pos / 10000 ** (i / embedding_size))
                    else:
                        pe[pos, i] = np.cos(
                            pos / 10000 ** ((i - 1) / embedding_size))
            self._positional_matrix = tf.constant(pe)
        return self._positional_matrix

    def call(self, in_):
        return tf.add(
            in_,
            self.calculate_positional_matrix(in_.shape[-1])
        )
