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
# import skimage
# from skimage import transform
# from skimage import data
import matplotlib.pyplot as plt
# from PIL import Image
import random
from tsc_data import load_data, TrainData, TestData



train = TrainData()
test = TestData()

BATCH_NUM = 4575
TAEGET_NUM = 62

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, TAEGET_NUM])
# weights W[784, TAEGET_NUM]   784=28*28
W = tf.Variable(tf.zeros([784, TAEGET_NUM]))
# biases b[TAEGET_NUM]
b = tf.Variable(tf.zeros([TAEGET_NUM]))

# flatten the images into a single line of pixels
# -1 in the shape definition means "the only possible dimension that will preserve the number of elements"
XX = tf.reshape(X, [-1, 784])

# The model
# Y = tf.nn.softmax(tf.matmul(XX, W) + b)
Y = tf.matmul(XX, W) + b
print(X, XX, Y)

# loss function: cross-entropy = - sum( Y_i * log(Yi) )
#                           Y: the computed output vector
#                           Y_: the desired output vector

# cross-entropy
# log takes the log of each element, * multiplies the tensors element by element
# reduce_mean will add all the components in the tensor
# so here we end up with the total cross-entropy for all images in the batch
# cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * BATCH_NUM * TAEGET_NUM * 1.0  # normalized for batches of 100 images,
                                                          # *10 because  "mean" included an unwanted division by 10

# cross-entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y))

# accuracy of the trained model, between 0 (worst) and 1 (best)
predict = tf.argmax(Y, 1)
correct_prediction = tf.equal(predict, tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training, learning rate = 0.003
train_step = tf.train.AdamOptimizer(0.003).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):
    batch_X, batch_Y = train.next_batch(BATCH_NUM)

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
    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y})

for i in range(200+1):
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
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = np.argmax(sample_labels[i])
    # print(truth, sample_labels[i])
    prediction = predicted[i]
    plt.subplot(5, 2, 1+i)
    plt.axis('off')
    color = 'green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
             fontsize=12, color=color)
    plt.imshow(sample_images[i].reshape(28,28))

plt.show()