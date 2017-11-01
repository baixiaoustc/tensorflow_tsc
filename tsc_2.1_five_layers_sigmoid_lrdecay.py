# -*- coding:utf-8 -*-
__author__ = 'baixiao'

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import argparse
import matplotlib.pyplot as plt
import random
import math
from tsc_data import load_data, TrainData, TestData
tf.set_random_seed(0)


train = TrainData()
test = TestData()

BATCH_NUM = 4575
TAEGET_NUM = 62

# neural network with 5 layers
#
# · · · · · · · · · ·          (input data, flattened pixels)       X [batch, 784]   # 784 = 28*28
# \x/x\x/x\x/x\x/x\x/       -- fully connected layer (sigmoid)      W1 [784, 200]      B1[200]
#  · · · · · · · · ·                                                Y1 [batch, 200]
#   \x/x\x/x\x/x\x/         -- fully connected layer (sigmoid)      W2 [200, 100]      B2[100]
#    · · · · · · ·                                                  Y2 [batch, 100]
#     \x/x\x/x\x/           -- fully connected layer (sigmoid)      W3 [100, 60]       B3[60]
#      · · · · ·                                                    Y3 [batch, 60]
#       \x/x\x/             -- fully connected layer (sigmoid)      W4 [60, 30]        B4[30]
#        · · ·                                                      Y4 [batch, 30]
#         \x/               -- fully connected layer (softmax)      W5 [30, 10]        B5[10]
#          ·                                                        Y5 [batch, 10]

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, TAEGET_NUM])
# variable learning rate
lr = tf.placeholder(tf.float32)

# five layers and their number of neurons (tha last layer has 10 softmax neurons)
L = 500
M = 250
# N = 150
# O = 100
# Weights initialised with small random values between -0.2 and +0.2
# When using RELUs, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones([K])/10
W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1))  # 784 = 28 * 28
B1 = tf.Variable(tf.zeros([L])/TAEGET_NUM)
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B2 = tf.Variable(tf.zeros([M])/TAEGET_NUM)
# W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
# B3 = tf.Variable(tf.zeros([N])/TAEGET_NUM)
# W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
# B4 = tf.Variable(tf.zeros([O])/TAEGET_NUM)
W5 = tf.Variable(tf.truncated_normal([M, TAEGET_NUM], stddev=0.1))
B5 = tf.Variable(tf.zeros([TAEGET_NUM]))

# The model
XX = tf.reshape(X, [-1, 784])
Y1 = tf.nn.sigmoid(tf.matmul(XX, W1) + B1)
Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)
# Y3 = tf.nn.relu6(tf.matmul(Y2, W3) + B3)
# Y4 = tf.nn.relu6(tf.matmul(Y3, W4) + B4)
Ylogits = tf.matmul(Y2, W5) + B5
Y = tf.nn.softmax(Ylogits)
print(X, XX, Y)

# loss function: cross-entropy = - sum( Y_i * log(Yi) )
#                           Y: the computed output vector
#                           Y_: the desired output vector
# cross-entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Ylogits))

# accuracy of the trained model, between 0 (worst) and 1 (best)
predict = tf.argmax(Ylogits, 1)
correct_prediction = tf.equal(predict, tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training step, learning rate will decay
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# You can call this function in a loop to train the model
def training_step(i, update_test_data, update_train_data):
    batch_X, batch_Y = train.next_batch(BATCH_NUM)

    # learning rate decay
    max_learning_rate = 0.002
    min_learning_rate = 0.001
    decay_speed = 1000.0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

    # compute training values for visualisation
    if update_train_data:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: batch_X, Y_: batch_Y})
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))

    # compute test values for visualisation
    if update_test_data:
        test_X, test_Y = test.next_batch()
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: test_X, Y_: test_Y})
        print(str(i) + ": ********* epoch " + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

    # the backpropagation training step
    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y, lr: learning_rate})

for i in range(1000+1):
    training_step(i, i % 100 == 0, i % 20 == 0)

# 运行模型
# Pick 10 random images
sample_indexes = random.sample(range(len(train.images)), 10)
print("sample_indexes", sample_indexes)
sample_images = [train.images[i] for i in sample_indexes]
sample_labels = [train.labels[i] for i in sample_indexes]

# Run the "predicted_labels" op.
predicted = sess.run([predict], feed_dict={X: sample_images})[0]

# Print the real and predicted labels
# print(sample_labels)
print("predicted", predicted)


# Display the predictions and the ground truth visually.
# fig = plt.figure(figsize=(10, 10))
# for i in range(len(sample_images)):
#     truth = np.argmax(sample_labels[i])
#     # print(truth, sample_labels[i])
#     prediction = predicted[i]
#     plt.subplot(5, 2, 1+i)
#     plt.axis('off')
#     color = 'green' if truth == prediction else 'red'
#     plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
#              fontsize=12, color=color)
#     plt.imshow(sample_images[i].reshape(28,28))
#
# plt.show()


# 0.002~0.0001, decay_speed = 1000.0     final 0.889 after 1k  with oscillation
# 0.002~0.0001, decay_speed = 800.0     final 0.884 after 1k  with oscillation
# 0.002~0.0001, decay_speed = 600.0     final 0.882 after 1k  with oscillation
# 0.002~0.0001, decay_speed = 400.0     final 0.873 after 1k  with oscillation
# 0.002~0.0001, decay_speed = 200.0     final 0.869 after 1k

# 0.003~0.0001, decay_speed = 1000.0     final 0.867 after 1k  with oscillation
# 0.003~0.0001, decay_speed = 200.0     final 0.868 after 1k  with oscillation

# 0.002~0.001, decay_speed = 1000.0     final 0.887 after 1k  with oscillation
# 0.002~0.001, decay_speed = 600.0     final 0.886 after 1k  with oscillation
# 0.002~0.001, decay_speed = 200.0     final 0.874 after 1k  with oscillation