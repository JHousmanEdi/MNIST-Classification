from __future__ import division, print_function
import os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier


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

ANN = MLPClassifier(hidden_layer_sizes = (250,250), activation = 'relu', max_iter = 100, alpha= 1e-4, solver='sgd',
                    learning_rate = 'constant', random_state=1,  learning_rate_init=0.001, verbose=True)

ANN.fit(X_train, y_train)

print("Training set test score: {}".format(ANN.score(X_train, y_train)))
print("Testing set accuracy: {}" .format(ANN.score(X_test, y_test)))
