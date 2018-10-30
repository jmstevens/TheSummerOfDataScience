import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# from tensorflow.contrib.layers import tf.layers.dense
from tensorflow.contrib.layers import dropout
from functools import partial
import numpy as np

n_inputs = 28*28
n_hidden_1 = 100
n_hidden_2 = 100
n_hidden_3 = 100
n_hidden_4 = 100
n_hidden_5 = 100
n_hidden_6 = 100
n_outputs = 10

tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name = 'X')
y = tf.placeholder(tf.int64, shape=(None), name = "y")

# training = tf.placeholder_with_default(False, shape=(), name='training')
norm_penalty = 0.001
learning_rate = 0.01
dropout_rate = 0.9
n_epochs = 1000
batch_size = 50

training = tf.placeholder_with_default(False, shape=(), name='training')
X_drop = dropout(X, dropout_rate, is_training=training)

my_dense_layer = partial(
    tf.layers.dense, activation=tf.nn.relu,
    kernel_regularizer=tf.contrib.layers.l2_regularizer(norm_penalty))

with tf.name_scope("dnn"):
    # arg_scope = tf.contrib.framework.arg_scope
    hidden1 = my_dense_layer(X_drop, n_hidden_1, name="hidden1")
    hidden1_drop = dropout(hidden1, dropout_rate, is_training=training)

    hidden2 = my_dense_layer(hidden1_drop, n_hidden_2,  name="hidden2")
    hidden2_drop = dropout(hidden2, dropout_rate, is_training=training)

    hidden3 = my_dense_layer(hidden2_drop, n_hidden_3, name="hidden3")
    hidden3_drop = dropout(hidden3, dropout_rate, is_training=training)

    hidden4 = my_dense_layer(hidden3_drop, n_hidden_4, name="hidden4")
    hidden4_drop = dropout(hidden3, dropout_rate, is_training=training)

    hidden5 = my_dense_layer(hidden4_drop, n_hidden_5, name="hidden5")
    hidden5_drop = dropout(hidden5, dropout_rate, is_training=training)

    hidden6 = my_dense_layer(hidden5_drop, n_hidden_6, name="hidden6")
    hidden6_drop = dropout(hidden6, dropout_rate, is_training=training)

    logits = my_dense_layer(hidden5_drop, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    base_loss = tf.reduce_mean(xentropy, name="avg_xentropy")
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([base_loss] + reg_losses, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

accuracy_summary = tf.summary.scalar('accuracy', accuracy)
accuracy_summary_test = tf.summary.scalar('accuracy_test', accuracy)
accuracy_summary_train = tf.summary.scalar('accuracy_train', accuracy)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())



with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", acc_valid)

        acc_train_str = accuracy_summary_train.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test_str = accuracy_summary_test.eval(feed_dict={X: X_valid, y: y_valid})
        file_writer.add_summary(acc_train_str, epoch)
        file_writer.add_summary(acc_test_str, epoch)

    save_path = saver.save(sess, "./my_model_final.ckpt")

#
# if __name__ == "__main__":
#   tf.app.run()
#   file_writer.close()
