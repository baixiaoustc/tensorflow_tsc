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
from PIL import Image
import random



ROOT_PATH = "/Users/baixiao/Go/src/github.com/baixiaoustc/tensorflow_pytest"
train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Training")
test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Testing")


# one channle
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(im_height, im_width, 1).astype(np.uint8)

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) for f in os.listdir(label_directory) if f.endswith(".ppm")]
        for f in file_names:
            # images.append(skimage.data.imread(f))
            imgfile = Image.open(f).convert('L')
            images.append(load_image_into_numpy_array(imgfile.resize((28, 28), Image.NEAREST)))
            label = np.zeros(62, dtype=np.int)
            label[int(d)] = 1
            # print label
            labels.append(label)
    return images, labels



class TrainData:
    _images = []
    _labels = []

    def __init__(self):
        self._images, self._labels = load_data(train_data_directory)
        print("TrainData", len(self._images), len(self._labels))

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def next_batch(self, limit=100):
        i = self._images
        l = self._labels
        return i, l


class TestData:
    _images = []
    _labels = []

    def __init__(self):
        self._images, self._labels = load_data(test_data_directory)
        print("TestData", len(self._images), len(self._labels))

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def next_batch(self, limit=100):
        i = self._images
        l = self._labels
        return i, l



# three channle
def load_image_into_numpy_array3(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(im_height, im_width, 3).astype(np.uint8)

def load_data3(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) for f in os.listdir(label_directory) if f.endswith(".ppm")]
        for f in file_names:
            # images.append(skimage.data.imread(f))
            imgfile = Image.open(f)
            images.append(load_image_into_numpy_array3(imgfile.resize((28, 28), Image.NEAREST)))
            label = np.zeros(62, dtype=np.int)
            label[int(d)] = 1
            # print label
            labels.append(label)
    return images, labels



class TrainData3:
    _images = []
    _labels = []

    def __init__(self):
        self._images, self._labels = load_data3(train_data_directory)
        print("TrainData", len(self._images), len(self._labels))

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def next_batch(self, limit=100):
        i = self._images
        l = self._labels
        return i, l


class TestData3:
    _images = []
    _labels = []

    def __init__(self):
        self._images, self._labels = load_data3(test_data_directory)
        print("TestData", len(self._images), len(self._labels))

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def next_batch(self, limit=100):
        i = self._images
        l = self._labels
        return i, l
'''
ROOT_PATH = "/Users/baixiao/Go/src/github.com/baixiaoustc/tensorflow_pytest"
train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Training")
test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Testing")

images, labels = load_data(train_data_directory)

images_array = np.array(images)
labels_array = np.array(labels)

# 定义模型
x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
y = tf.placeholder(dtype=tf.int64, shape=[None])
images_flat = tf.contrib.layers.flatten(x)
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_pred = tf.equal(tf.argmax(logits, 1), y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", correct_pred)


# 训练模型
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(2001):
        # print('EPOCH', i)
        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images, y: labels})
        if i % 10 == 0:
            print("accuracy_val: ", accuracy_val)
        # print('DONE WITH EPOCH')


# 运行模型
# Pick 10 random images
sample_indexes = random.sample(range(len(images)), 10)
sample_images = [images[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

# Run the "predicted_labels" op.
predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]

# Print the real and predicted labels
print(sample_labels)
print(predicted)


# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
             fontsize=12, color=color)
    plt.imshow(sample_images[i])

plt.show()
'''