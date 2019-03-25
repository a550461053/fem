# coding=utf-8

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime

# parameters
BATCH_START = 0  # start index of batch_data
TIME_STEPS = 10  # backpropagation through time
BATCH_SIZE = 40  # batch size 30
INPUT_SIZE = 12  # input data size of nn
OUTPUT_SIZE = 1  # output data size of nn
CELL_SIZE = 10   # hidden unit size of RNN cell
learning_rate = 0.006  # learning rate
ROUND = 1000      # train rounds
shuffle = False  # shuffle data

def getTIME_COUNT(data_size, BATCH_SIZE, TIME_STEPS):
    # get the time_count
    # time_count = (data_size - batch_size*time_steps) / time_steps
    # (4000 - 40*10) / 10 = 360
    # such as: 0-400, 10-410, 20-420, ... , 3600-4000
    return int((data_size - BATCH_SIZE * TIME_STEPS) / TIME_STEPS)

TIME_COUNT = getTIME_COUNT(4000, BATCH_SIZE, TIME_STEPS)  # (4000 - BATCH_SIZE * TIME_STEPS) / TIME_STEPS  # 20

print("BATCH_SIZE", BATCH_SIZE, "TIME_STEP", TIME_STEPS, "TIME_COUNT", TIME_COUNT)

# input data path
input_data_path = "./data/input_data.csv"
# load data to DataFrame
df_read = pd.read_csv(input_data_path, header=None, names=['number', 'BB0', 'BB1', 'F0', 'F1', 'F2', 'F3', 'HH0', 'HH1', 'P0', 'P1', 'P2', 'P3'])

# log path
TIMESTAMPE = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
train_log_dir = "logs/rnn/train/" + TIMESTAMPE
test_log_dir = "logs/rnn/test/" + TIMESTAMPE

# get input data DateFrame
x = df_read[['BB0', 'BB1', 'F0', 'F1', 'F2', 'F3', 'HH0', 'HH1', 'P0', 'P1', 'P2', 'P3']]
y = df_read['number']

# Standardize
ss_x = preprocessing.StandardScaler()
train_x = ss_x.fit_transform(x)
ss_y = preprocessing.StandardScaler()
train_y = ss_y.fit_transform(y.reshape(-1, 1))
# train_y = y.values.reshape(-1, 1)

if shuffle:
    # when regression, set shuffle to False
    train_x_disorder, test_x_disorder, train_y_disorder, test_y_disorder = train_test_split(train_x, train_y,
                                                                                            train_size=0.8, random_state=33)
    del train_x
    del train_y
    train_x = train_x_disorder
    train_y = train_y_disorder
    test_x = test_x_disorder
    test_y = test_y_disorder
else:
    # split data to train 0-4000, test 4000-4540
    test_x = train_x[4000:]  # 10*40batch
    test_y = train_y[4000:]
    train_x = train_x[:4000]
    train_y = train_y[:4000]

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)  # (4541, 12)


# get batch data when train.
def get_batch_train():
    global train_x, train_y, BATCH_START, TIME_STEPS
    x_part1 = train_x[BATCH_START: BATCH_START + TIME_STEPS * BATCH_SIZE]
    y_part1 = train_y[BATCH_START: BATCH_START + TIME_STEPS * BATCH_SIZE]
    # print('period = ', BATCH_START, BATCH_START + TIME_STEPS * BATCH_SIZE)

    seq = x_part1.reshape((BATCH_SIZE, TIME_STEPS, INPUT_SIZE))
    res = y_part1.reshape((BATCH_SIZE, TIME_STEPS, 1))

    BATCH_START += TIME_STEPS

    # returned seq, res and xs: shape (batch, step, input)
    return [seq, res]


# get batch data when test.
def get_batch_test():
    global train_x, train_y, BATCH_START, TIME_STEPS
    x_part1 = test_x[BATCH_START: BATCH_START + TIME_STEPS * BATCH_SIZE]
    y_part1 = test_y[BATCH_START: BATCH_START + TIME_STEPS * BATCH_SIZE]
    print('period = ', BATCH_START, BATCH_START + TIME_STEPS * BATCH_SIZE)

    seq = x_part1.reshape((BATCH_SIZE, TIME_STEPS, INPUT_SIZE))
    res = y_part1.reshape((BATCH_SIZE, TIME_STEPS, 1))

    BATCH_START += TIME_STEPS

    # returned seq, res and xs: shape (batch, step, input)
    return [seq, res]


def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    xs = np.arange(BATCH_START, BATCH_START + TIME_STEPS * BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10 * np.pi)
    print('xs.shape=', xs.shape)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS

    # returned seq, res and xs: shape (batch, step, input)
    # np.newaxis add a dim to 3D, xs is save last batch status
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]


class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        '''
        :param n_steps: steps time of each batch data 
        :param input_size: input size 
        :param output_size: output size 
        :param cell_size: cell size 
        :param batch_size: batch size 
        '''
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')  # xs 3D
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')  # ys 3D
        # in_hidden layer
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        # use LSTM cell
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        # out_hidden layer
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        # compute loss
        with tf.name_scope('cost'):
            self.compute_cost()
        # train operation
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

    # add input layer
    def add_input_layer(self, ):
        # l_in_x:(batch*n_step, in_size)
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # -1 表示任意行数
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.cell_size, ])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

    # LSTM cell
    def add_cell(self):
        # lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        # time_major=False 表示时间主线不是第一列batch
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)

    # add output layer
    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            # precition result
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out

    # compute loss
    def compute_cost(self):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size,
                name='average_cost')
            tf.summary.scalar('cost', self.cost)

    # (y_pre - y_target)**2
    def ms_error(self, y_pre, y_target):
        # return tf.square(tf.sub(y_pre, y_target))
        return tf.square(tf.subtract(y_pre, y_target))

    # define weight
    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1., )
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    # define bias
    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)


if __name__ == '__main__':
    seq, res = get_batch_train()
    # Instantiated model
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)

    # setup tf network
    sess = tf.Session()
    merged = tf.summary.merge_all()

    # define writer logs
    train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    test_writer = tf.summary.FileWriter(test_log_dir)

    sess.run(tf.global_variables_initializer())

    # training
    for j in range(ROUND):
        pred_res = None
        # TIME_COUNT period
        for i in range(TIME_COUNT):
            seq, res = get_batch_train()

            if i == 0:
                # when initial state
                feed_dict = {
                    model.xs: seq,
                    model.ys: res,
                    # create initial state
                }
            else:
                feed_dict = {
                    model.xs: seq,
                    model.ys: res,
                    model.cell_init_state: state  # use last state as the initial state for this run
                }

            # all operation run
            _, cost, state, pred_train = sess.run(
                [model.train_op, model.cost, model.cell_final_state, model.pred],
                feed_dict=feed_dict)
            pred_res = pred_train

            summary = sess.run(merged, feed_dict)
            # !range is: i*j, or it will overlapping in tensorboard
            train_writer.add_summary(summary, i * j)

        print('{0} train cost: '.format(j), round(cost, 4))
        BATCH_START = 0  # reset start index of batch

    # test
    print("testing...")
    BATCH_START = 0  # reset start index of batch
    pred_res = None
    # BATCH_SIZE = 30
    # TIME_STEPS = 10
    # TIME_COUNT = getTIME_COUNT(540, BATCH_SIZE, TIME_STEPS)
    TIME_COUNT = int((540 - BATCH_SIZE * TIME_STEPS) / TIME_STEPS)
    print("BATCH_SIZE", BATCH_SIZE, "TIME_STEP", TIME_STEPS, "TIME_COUNT", TIME_COUNT)
    for i in range(TIME_COUNT):
        seq, res = get_batch_test()

        if i == 0:
            feed_dict = {
                model.xs: seq,
                model.ys: res,
                # create initial state
            }
        else:
            feed_dict = {
                model.xs: seq,
                model.ys: res,
                model.cell_init_state: state  # use last state as the initial state for this run
            }

        _, cost, state, pred_test = sess.run(
            [model.train_op, model.cost, model.cell_final_state, model.pred],
            feed_dict=feed_dict)
        pred_res = pred_test

        summary = sess.run(merged, feed_dict)
        test_writer.add_summary(summary, i)

    print('{0} test cost: '.format(j), round(cost, 4))

    # r_size = BATCH_SIZE * TIME_STEPS
    ################################################################################
    # plot
    ################################################################################
    # get one batch data to plot
    train_y = train_y[3590:3990]
    test_y = test_y[130: 530]
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(20, 3))
    axes1 = fig.add_subplot(2, 1, 1)
    # show 300 row data
    line1, = axes1.plot(range(300), pred_train.flatten()[-300:], 'b--', label='train rnn result')
    line3, = axes1.plot(range(300), train_y.flatten()[- 300:], 'g', label='train real')
    axes1.grid()

    axes2 = fig.add_subplot(2, 1, 2)
    # show 300 row data
    line11, = axes2.plot(range(300), pred_test.flatten()[-300:], 'r--', label='test rnn result')
    line33, = axes2.plot(range(300), test_y.flatten()[- 300:], 'y', label='test real')
    axes2.grid()


    fig.tight_layout()
    plt.legend(handles=[line1, line3, line11, line33])
    plt.title('RNN FEM')
    plt.show()


    # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
    # $ tensorboard --logdir='logs'