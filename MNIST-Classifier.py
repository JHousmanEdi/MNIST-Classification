import tensorflow as tf
from __future__ import division, print_function
import os
import numpy as np
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
train_data_path = os.path.join(os.getcwd(), 'train.csv')
test_data_path = os.path.join(os.getcwd(), 'test.csv')
#Import training set
MNIST = pd.read_csv(train_data_path, delimiter=',')
print(MNIST.head(5))
#Seperate labels and values for training and testing
X = MNIST.drop('label', axis = 1)
y = MNIST.ix[:,0]
X_train = X.iloc[:30000, :]
y_train = y.iloc[:30000]
X_test = X.iloc[30000:42000, :]
y_test = y.iloc[30000:42000]




print('Number of instances: {}, number of features: {}'.format(X_train.shape[0], X_train.shape[1]))

learning_rate = 0.001
training_iters = 1000
batch_size = 100
display_step = 2

n_input = 784
n_classes = 10
dropout = 0.75


x = tf.placeholder(tf.float32, [None, n_input]) #number of features in placeholder
y = tf.placeholder(tf.float32, [None, n_classes]) #Number of classes in placeholder
kept_probability = tf.placeholder(tf.float32)


def conv2d(x, W, b, strides = 1):
    x = tf.nn.conv2d(x, W, strides = [1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def convolutional_net_predict(input, parameters, probability):
    x = tf.reshape(input, shape=[-1, 28, 28, 1])

    #conv layer:
    conv1layer = conv2d(input, parameters['wc1'], parameters['bc2'])
    convlayer2 = conv2d(conv1layer, parameters['wc2'], parameters['bc2'])
    conv1layer = maxpool2d(conv1layer, k =2)
    convlayer2 = maxpool2d(convlayer2, k=2)

    fc1 = tf.reshape(convlayer2, [-1, parameters['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, parameters['wd1']), parameters['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, probability)

    out = tf.add(tf.matmul(fc1, parameters['out']), parameters['out'])
    return out
   #Parameters for the convolutional net, all weights and biases.
parameters = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2':tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    'wout': tf.Variable(tf.random_normal([1024, n_classes])),
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1':tf.Variable(tf.random_normal([1024])),
    'bout': tf.Variable(tf.random_normal([n_classes]))

}
MNIST_Predictor = convolutional_net_predict(x, parameters, kept_probability)

cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=MNIST_Predictor, labels = y))
optimization = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost_func)

correct_pred = tf.equal(tf.argmax(MNIST_Predictor, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    step = 1
    while step * batch_size < training_iters:
        X_train, y_train = int(mnist.train.num_examples/batch_size)
        session.run(optimization, feed_dict={x: X_train, y: y_train, kept_probability: kept_probability})

        if step % display_step == 0:
            loss, acc = session.run([cost_func, accuracy], feed_dict={x: X_train, y: y_train, kept_probability: 1.})

            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            step += 1
            print("Training done")

            print("Testing Accuracy:", \
                  session.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, kept_probability: 1.}))
