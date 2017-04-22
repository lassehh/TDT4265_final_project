#import load_traffic_signs
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from keras import backend as K
from keras.utils import np_utils
K.set_image_dim_ordering('th')


#[X, Y] = load_traffic_signs.training('Data/Final_Training/Images')
#X = np.array(X).astype('float32')

# read images
print('Loading data as numpy array of arrays...')
X_train = np.load('X_train_data.npy')
X_test = np.load('X_test_data.npy')
Y_train = np.load('Y_train_data.npy')
Y_train = np_utils.to_categorical(Y_train)
Y_test = np.load('Y_test_data.npy')
Y_test = np_utils.to_categorical(Y_test)
print('Done!')

#plt.imshow(X_test[42])
#plt.show()

#img = Image.open("someimg.png")
#print(len(X_test[42][0]), len(X_test[42][1][20]))


#img /= np.std(img)
#print(np.std(img, axis = 0))
#print(np.linalg.norm(img[9,9,1]))
#print(img[9,9])
#img -= np.mean(img)
# img = img.convert("RGB", rgb2bgr)
#img = img.resize((20,20), Image.ANTIALIAS)
img = X_train[0]
print(img.size)
plt.imshow(img)
plt.show()