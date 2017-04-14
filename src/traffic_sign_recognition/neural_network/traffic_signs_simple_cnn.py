import matplotlib.pyplot as plt
import numpy as np
import csv
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from read_traffic_signs import *
K.set_image_dim_ordering('th')

# read images
[X_train, Y_train] = readTrainingTrafficSigns('GTSRB/Final_Training/Images')
[X_test, Y_test] = readTestingTrafficSigns('GTSRB/Final_Testing/Images')

# format, normalize
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
X_train = X_train / 255
X_test = X_test / 255

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
num_classes = Y_test.shape[1]

# show a single image
plt.imshow(X_train[22])
print(X_train[22][0][0])
plt.show()

def baseline_model(): # create model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(1, 28, 28)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build the model
model = baseline_model()

# Fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=200, verbose=2)
model.save('traffic_signs_simple_cnn.h5')
json_string = model.to_json()

# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))