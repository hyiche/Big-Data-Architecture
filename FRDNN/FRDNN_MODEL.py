import tensorflow as tf
import numpy as np
import cv2

class Model(object):
    def __init__(self, total_data):
        self.total_data = total_data
        self.fuzzyIn = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="FuzzyIn")
        self.labelIn = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="LabelIn")
        self.outputs = None
        self.loss = None

        with tf.name_scope("k-means"):
            total_data = np.asarray(a=self.total_data, dtype=np.float32)
            # setting for k-means
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
            flags = cv2.KMEANS_RANDOM_CENTERS
            compactness, label_kmeans, centers = cv2.kmeans(
                data=total_data, K=3, bestLabels=None, criteria=criteria, attempts=10, flags=flags)
            List0 = []
            List1 = []
            List2 = []
            for ii in range(0, total_data.size):
                if 0 == label_kmeans[ii][0]:
                    List0.append(total_data[ii][0])
                if 1 == label_kmeans[ii][0]:
                    List1.append(total_data[ii][0])
                if 2 == label_kmeans[ii][0]:
                    List2.append(total_data[ii][0])
            data0_tensor = tf.convert_to_tensor(value=List0, dtype=tf.float32)
            data1_tensor = tf.convert_to_tensor(value=List1, dtype=tf.float32)
            data2_tensor = tf.convert_to_tensor(value=List2, dtype=tf.float32)
            mean0, variance0 = tf.nn.moments(x=data0_tensor, axes=[0])
            mean1, variance1 = tf.nn.moments(x=data1_tensor, axes=[0])
            mean2, variance2 = tf.nn.moments(x=data2_tensor, axes=[0])

        with tf.name_scope("fuzzy-layer"):
            fuzzy0 = tf.exp(tf.negative(tf.nn.batch_normalization(x=self.fuzzyIn, mean=mean0,
                    variance=variance0, offset=None, scale=None, variance_epsilon=0.001)))
            fuzzy1 = tf.exp(tf.negative(tf.nn.batch_normalization(x=self.fuzzyIn, mean=mean1,
                    variance=variance1, offset=None, scale=None, variance_epsilon=0.001)))
            fuzzy2 = tf.exp(tf.negative(tf.nn.batch_normalization(x=self.fuzzyIn, mean=mean2,
                    variance=variance2, offset=None, scale=None, variance_epsilon=0.001)))
            fuzzyOut = tf.concat(values=[fuzzy0, fuzzy1, fuzzy2], axis=0, name="FuzzyOut")
            fuzzyOut = tf.reshape(tensor=fuzzyOut, shape=[1, 150])

        with tf.name_scope("MLP-layer"):
            dense1 = tf.layers.dense(inputs=fuzzyOut, units=150, activation=tf.nn.sigmoid)
            dense2 = tf.layers.dense(inputs=dense1, units=100, activation=tf.nn.sigmoid)
            dense3 = tf.layers.dense(inputs=dense2, units=50, activation=tf.nn.sigmoid)
            dense4 = tf.layers.dense(inputs=dense3, units=20)
            # decoder1 = tf.layers.dense(inputs=dense4, units=50, activation=tf.nn.sigmoid)
            # decoder2 = tf.layers.dense(inputs=decoder1, units=100, activation=tf.nn.sigmoid)
            # decoder3 = tf.layers.dense(inputs=decoder2, units=150, activation=tf.nn.sigmoid)
            # loss3 = tf.losses.mean_squared_error(labels=dense3, predictions=decoder1)
            # loss2 = tf.losses.mean_squared_error(labels=dense2, predictions=decoder2)
            # loss1 = tf.losses.mean_squared_error(labels=dense1, predictions=decoder3)
            # train_Autoencoder = tf.train.AdamOptimizer(0.002).minimize(loss1 + loss2 + loss3)
            # optimizer = tf.train.AdamOptimizer(0.002)
            # train_op = optimizer.minimize(loss1 + loss2 + loss3)

        with tf.name_scope("RNN-layer"):
            # vanilla_rnn_layer
            rnn_In = tf.reshape(tensor=dense4, shape=[1, 20, 1])
            rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=1)
            initial_state = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
            outputs, state = tf.nn.dynamic_rnn(rnn_cell, rnn_In, initial_state=initial_state, dtype=tf.float32, time_major=False)
            self.outputs = tf.reshape(outputs, [20, 1])

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.pow(self.outputs - self.labelIn, 2))