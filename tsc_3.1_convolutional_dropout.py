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

# neural network structure for this sample:
#
# · · · · · · · · · ·      (input data, 1-deep)                 X [batch, 28, 28, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer 5x5x1=>4 stride 1        W1 [5, 5, 1, 4]        B1 [4]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                           Y1 [batch, 28, 28, 4]
#   @ @ @ @ @ @ @ @     -- conv. layer 5x5x4=>8 stride 2        W2 [5, 5, 4, 8]        B2 [8]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                             Y2 [batch, 14, 14, 8]
#     @ @ @ @ @ @       -- conv. layer 4x4x8=>12 stride 2       W3 [4, 4, 8, 12]       B3 [12]
#     ∶∶∶∶∶∶∶∶∶∶∶                                               Y3 [batch, 7, 7, 12] => reshaped to YY [batch, 7*7*12]
#      \x/x\x\x/        -- fully connected layer (relu)         W4 [7*7*12, 200]       B4 [200]
#       · · · ·                                                 Y4 [batch, 200]
#       \x/x\x/         -- fully connected layer (softmax)      W5 [200, 10]           B5 [10]
#        · · ·                                                  Y [batch, 10]

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, TAEGET_NUM])
# variable learning rate
lr = tf.placeholder(tf.float32)
# Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
pkeep = tf.placeholder(tf.float32)

# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
K = 6  # first convolutional layer output depth
L = 12  # second convolutional layer output depth
M = 24  # third convolutional layer
N = 300  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))  # 5x5 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.ones([K])/TAEGET_NUM)
W2 = tf.Variable(tf.truncated_normal([4, 4, K, L], stddev=0.1))
B2 = tf.Variable(tf.ones([L])/TAEGET_NUM)
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.ones([M])/TAEGET_NUM)

W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
B4 = tf.Variable(tf.ones([N])/TAEGET_NUM)
W5 = tf.Variable(tf.truncated_normal([N, TAEGET_NUM], stddev=0.1))
B5 = tf.Variable(tf.ones([TAEGET_NUM]))

# The model
stride = 1  # output is 28x28
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2  # output is 14x14
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2  # output is 7x7
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

Y4 = tf.nn.sigmoid(tf.matmul(YY, W4) + B4)
YY4 = tf.nn.dropout(Y4, pkeep)
Ylogits = tf.matmul(YY4, W5) + B5
Y = tf.nn.softmax(Ylogits)

# loss function: cross-entropy = - sum( Y_i * log(Yi) )
#                           Y: the computed output vector
#                           Y_: the desired output vector
# cross-entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Ylogits))

# accuracy of the trained model, between 0 (worst) and 1 (best)
predict = tf.argmax(Ylogits, 1)
correct_prediction = tf.equal(predict, tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training step, learning rate = 0.003
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):
    batch_X, batch_Y = train.next_batch(BATCH_NUM)

    # learning rate decay
    max_learning_rate = 0.002
    min_learning_rate = 0.0001
    decay_speed = 200.0  # 0.003-0.0001-2000=>0.9826 done in 5000 iterations
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

    # compute training values for visualisation
    if update_train_data:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: batch_X, Y_: batch_Y, pkeep: 1.0})
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))

    # compute test values for visualisation
    if update_test_data:
        test_X, test_Y = test.next_batch()
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: test_X, Y_: test_Y, pkeep: 1.0})
        print(str(i) + ": ********* epoch " + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

    # the backpropagation training step
    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y, lr: learning_rate, pkeep: 0.75})

for i in range(200+1):
    training_step(i, i % 100 == 0, i % 20 == 0)

# 运行模型
# Pick 10 random images
sample_indexes = random.sample(range(len(train.images)), 10)
print("sample_indexes", sample_indexes)
sample_images = [train.images[i] for i in sample_indexes]
sample_labels = [train.labels[i] for i in sample_indexes]

# Run the "predicted_labels" op.
predicted = sess.run([predict], feed_dict={X: sample_images, pkeep: 1.0})[0]

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


# layers 4 8 12 300, patches 5x5str1 5x5str2 4x4str2 dropout=0.75 best 0.887 after 200 iterations/ best 0.891 after 1000 iterations
# layers 6 12 24 300, patches 5x5str1 4x4str2 4x4str2 dropout=0.75 best 0.887 after 200 iterations/ best  after 1000 iterations