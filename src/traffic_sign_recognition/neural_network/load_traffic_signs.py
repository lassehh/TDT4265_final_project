import csv
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
from PIL import Image


# This module only works for the specific case of reading traffic signs and is not
# intended as a general file reading module. The raw pictures/folders must lie in the
# roothpath path.

def load_training_data(rootpath):
    X_train = []
    Y_train = []
    i = 0
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/'
        gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')
        gtReader = csv.reader(gtFile, delimiter=';')
        gtReader.__next__()
        for row in gtReader:
            X_train.append(Image.open(prefix + row[0]))
            X_train[i] = np.array(X_train[i].resize((32, 32), Image.ANTIALIAS))
            #X_train[i] -= np.mean(X_train[i])
            Y_train.append(row[7])
            i += 1
        gtFile.close()
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    X_train = X_train.reshape(X_train.shape[0], 3, 32, 32)
    #X_train /= 255
    print('Training data loaded with X_train.shape: ', X_train.shape)
    np.save(file = 'data_numpy/X_train_data', arr = X_train)
    np.save(file = 'data_numpy/Y_train_data', arr = Y_train)

def load_testing_data(rootpath):
    X_test = []
    Y_test = []
    i = 0
    prefix = rootpath + '/'
    gtFile = open(prefix + 'GT-final_test' + '.csv')
    gtReader = csv.reader(gtFile, delimiter=';')
    gtReader.__next__()
    for row in gtReader:
        X_test.append(Image.open(prefix + row[0]))
        X_test[i] = np.array(X_test[i].resize((32, 32), Image.ANTIALIAS))
        #X_test[i] -= np.mean(X_test[i])
        Y_test.append(row[7])
        i += 1
    gtFile.close()
    X_test = np.asarray(X_test)
    Y_test = np.asarray(Y_test)
    X_test = X_test.reshape(X_test.shape[0], 3, 32, 32)
    #X_test /= 255
    print('Testing data loaded with X_test.shape: ', X_test.shape)
    np.save(file='data_numpy/X_test_data', arr=X_test)
    np.save(file='data_numpy/Y_test_data', arr=Y_test)


load_training_data('data/Final_Training/Images')
load_testing_data('data/Final_Testing/Images')