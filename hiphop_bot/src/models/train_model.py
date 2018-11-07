import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

# class RNNModel:
#     def __init__(self,
#                  vocabularly_size,
#                  batch_size,
#                  sequence_length,
#                  hidden_layer_size,
#                  cells_size,
#                  gradient_clip=5,
#                  training=True):
#
#     cells=[]
#     [cells.append(rnn.LSTMCell(hidden_layer_size)) for _ in range(cells_size)]
#     self.cell = rnn.MultiRNNCell(cells)
#
#     self.input_data = tf.placeholder(tf.int32, [batch_size, sequence_length])
#     self.targets = tf.placeholder(tf.int32, [batch_size, sequence_length])
#     self.initial_state = self.cell.zero_state(batch_size, tf.float32)
