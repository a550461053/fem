# coding=utf-8

import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import datetime


# parameters
dropout = 0.7
learning_rate = 0.05  # 如果输出归一化，则lr需要降低，否则发散
ROUND = 300
standardY = True  # 归一化输出

# input data path
input_data_path = "./data/input_data.csv"
# load data to DataFrame
df_read = pd.read_csv(input_data_path, header=None, names=['number', 'BB0', 'BB1', 'F0', 'F1', 'F2', 'F3', 'HH0', 'HH1', 'P0', 'P1', 'P2', 'P3'])

# log path
TIMESTAMPE = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
train_log_dir = "logs/cnn/train/" + TIMESTAMPE
test_log_dir = "logs/cnn/test/" + TIMESTAMPE

# get input data DateFrame
x = df_read[['BB0', 'BB1', 'F0', 'F1', 'F2', 'F3', 'HH0', 'HH1', 'P0', 'P1', 'P2', 'P3']]
y = df_read['number']

# data preprocess: padding column
# get data from 2 to 6 column
x_4 = x.ix[:, 2: 6]
# padding x_4 to x, then x has 16 columns and x can reshape to 4*4
# it comes to a square
x = np.column_stack([x, x_4])

print('##################################################################')

# Standardize
ss_x = preprocessing.StandardScaler()
x = ss_x.fit_transform(x)

if standardY:
    ss_y = preprocessing.StandardScaler()
    y = ss_y.fit_transform(y.values.reshape(-1, 1))

# data split: data random select
train_x_disorder, test_x_disorder, train_y_disorder, test_y_disorder = train_test_split(x, y,
                                                                                        train_size=0.8, random_state=33)

if standardY:
    pass
    # ss_y = preprocessing.StandardScaler()
    # train_y_disorder = ss_y.fit_transform(train_y_disorder.values.reshape(-1, 1))
    # test_y_disorder = ss_y.transform(test_y_disorder.values.reshape(-1, 1))
else:
    # all thing must be a 2D matrix in new version tf, or reshape to 2D
    train_y_disorder = train_y_disorder.values.reshape(-1, 1)
    test_y_disorder = test_y_disorder.values.reshape(-1, 1)


##############################################################################
# define network
##############################################################################

# compute the accuracy of nn
# def compute_accuracy(v_xs, v_ys):
#     global prediction
#     y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
#     correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
#     return result


# define Variable storage
def variable_summary(var, name):
    with tf.name_scope("summary"):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean/" + name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev/" + name, stddev)

# define weight matrix
def weight_variable(shape, layer_name):
    # with tf.name_scope("weights"):
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    variable_summary(weights, layer_name + "/weights")
    return weights


# define biases
def bias_variable(shape, layer_name):
    # with tf.name_scope("biases"):
    biases = tf.Variable(tf.constant(0.1, shape=shape))
    variable_summary(biases, layer_name + "/biases")
    return biases


# define convolution network
def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1] x_movement、y_movement is step
    # Must have strides[0] = strides[3] = 1 padding='SAME' means not change shape
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# define max_pooling
def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# define network layer ! —— not used
def new_layer(input_tensor, input_dim, out_dim, layer_name, act_funcion=tf.nn.relu):
    with tf.name_scope(layer_name):
        weights = weight_variable([2, 2, input_dim, out_dim])


# define placeholder for inputs to network
with tf.name_scope("input"):
    xs = tf.placeholder(tf.float32, [None, 16])
    ys = tf.placeholder(tf.float32, [None, 1])
    keep_prob = tf.placeholder(tf.float32)

    # raw data 16*1 -> 4*4
    x_image = tf.reshape(xs, [-1, 4, 4, 1])

# conv1 layer
layer_name = "conv1"
with tf.name_scope(layer_name):
    # patch 2x2, in size 1, out size 32, it is deeping
    W_conv1 = weight_variable([2, 2, 1, 32], layer_name)
    b_conv1 = bias_variable([32], layer_name)
    # output size 2x2x32
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # h_pool1 = max_pool_2x2(h_conv1)     # output size 2x2x32
    # save histogram of output in this layer
    tf.summary.histogram(layer_name + "/activations", h_conv1)

## conv2 layer
layer_name = "conv2"
with tf.name_scope(layer_name):
    # patch 2x2, in size 32, out size 64
    W_conv2 = weight_variable([2, 2, 32, 64], layer_name)
    b_conv2 = bias_variable([64], layer_name)
    # output shape 4*4*64
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
    # save histogram of output in this layer
    tf.summary.histogram(layer_name + "/activations", h_conv2)

# full connection
## fc1 layer
layer_name = "fc1"
with tf.name_scope(layer_name):
    # reduction dimension: 4*4*64 -> 512 * 1
    W_fc1 = weight_variable([4 * 4 * 64, 512], layer_name)
    b_fc1 = bias_variable([512], layer_name)

    # 4*4*64 reshape to 512*1
    h_pool2_flat = tf.reshape(h_conv2, [-1, 4 * 4 * 64])
    # Activation function: relu
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # save histogram of output nn
    tf.summary.histogram(layer_name + "/activations", h_fc1_drop)

# full connection layer
## fc2 layer ## full connection
layer_name = "fc2"
with tf.name_scope(layer_name):
    # 512 * 1 -> 1*1
    W_fc2 = weight_variable([512, 1], layer_name)
    b_fc2 = bias_variable([1], layer_name)

    # prediction result of nn
    prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    # prediction = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # save histogram of output in this layer
    # tf.summary.histogram(layer_name + "/activations", prediction)

# loss = MSE (prediction - y)
with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    # save cross_entropy
    tf.summary.scalar("cross_entropy", cross_entropy)

# Optimizer, lr = 0.01
optimize = tf.train.AdamOptimizer(learning_rate)
with tf.name_scope("train"):
    train_step = optimize.minimize(cross_entropy)

##############################################################################
# init
##############################################################################
# set network
sess = tf.Session()
init = tf.global_variables_initializer()

# merge all summary and operation
merged = tf.summary.merge_all()
# define train_writer and test_writer
train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
test_writer = tf.summary.FileWriter(test_log_dir)

# init all variables
sess.run(init)


##############################################################################
# start training
##############################################################################

def feed_dict(train):
    """
    Make a TensorFlow feed_dict: maps data onto Tensor placeholders.
    """
    if train:
        # feed to train
        x = train_x_disorder
        y = train_y_disorder
        k = dropout
    else:
        # feed to test
        x = test_x_disorder
        y = test_y_disorder
        k = 1.0
    return {xs: x, ys: y, keep_prob: k}

for i in range(ROUND):
    if i % 10 == 0:
        # summary and cross_entropy of test
        summary, test_cross_entropy = sess.run(
            [merged, cross_entropy],
            feed_dict=feed_dict(False)
        )
        test_writer.add_summary(summary, i)
        print("test_cross_entropy at step %s : %s" % (i, test_cross_entropy))

    if i % 100 == 99:
        # Record execution stats
        # save summary and cross_entropy of train
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _, train_cross_entropy = sess.run(
            [merged, train_step, cross_entropy],
            feed_dict=feed_dict(True),
            options=run_options,
            run_metadata=run_metadata
        )  # {xs: train_x_disorder, ys: train_y_disorder, keep_prob: 0.7})
        train_writer.add_run_metadata(run_metadata, "step%03d" % i)
        train_writer.add_summary(summary, i)
        print("adding run metadata for", i)
    else:
        # Record a summary
        summary, _ = sess.run(
            [merged, train_step],
            feed_dict=feed_dict(True)
        )
        train_writer.add_summary(summary, i)

# get prediction value of test data
prediction_value = sess.run(prediction, feed_dict={xs: test_x_disorder, ys: test_y_disorder, keep_prob: 1.0})
##############################################################################
# plot
##############################################################################
import matplotlib.pyplot as plt

# define figure
fig = plt.figure(figsize=(20, 3))
# subplot 2, set axes1 to position 1
axes1 = fig.add_subplot(2, 1, 1)
# line1 is cnn data
line1, = axes1.plot(range(len(prediction_value)), prediction_value, 'b--', label='cnn', linewidth=2)
line2, = axes1.plot(range(len(test_y_disorder)), test_y_disorder, 'g', label='real')

axes1.grid()

# if standardY, then inverse transform output y
if standardY:
    print("inverse transform output y...")
    prediction_value = ss_y.inverse_transform(prediction_value)
    test_y_disorder = ss_y.inverse_transform(test_y_disorder)
axes2 = fig.add_subplot(2, 1, 2)
# line11 is raw-cnn data
line11, = axes2.plot(range(len(prediction_value)), prediction_value, 'r--', label='raw-cnn', linewidth=2)
line22, = axes2.plot(range(len(test_y_disorder)), test_y_disorder, 'y', label='raw-real')
axes2.grid()

fig.tight_layout()
plt.legend(handles=[line1, line2, line11, line22])
plt.title('CNN FEM')
plt.show()
