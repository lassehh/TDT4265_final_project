import sys
sys.path.append('../')
from traffic_sign_recognition.neural_network import load_traffic_signs

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import load_model
from keras.models import Sequential
from keras import backend as K
K.set_image_dim_ordering('th')

#TODO: The convnet returns unreal high probabilities for many of the traffic signs even thought it's looking
#TODO: at some random place in the image. Especially high for white regions.
#TODO: Is the convnet wrong, or the script?
#TODO: The same preproccesing for the training is applied to the data.


def scan_image(np_image, strides, convnet):

    (height, width) = (np_image.shape[0], np_image.shape[1])
    print(width,height)
    (x_ptr, y_ptr) = (0, 0)
    highest_pred = 0
    highest_pred_pos = (0,0)
    highest_pred_filter = 0

    for filter in [128]:
        while y_ptr + filter < height:
            while x_ptr + filter < width:

                np_image_slice = np_image[y_ptr:filter+y_ptr:1, x_ptr:filter+x_ptr:1]
                image_slice = Image.fromarray(np_image_slice, 'RGB')
                cnn_input = np.array(image_slice.resize((32, 32), Image.ANTIALIAS))
                cnn_input = cnn_input.reshape(1,3,32,32).astype('float32')
                cnn_input /= 255
                cnn_input -= np.mean(cnn_input)


                predictions = convnet.predict(cnn_input, batch_size= 1, verbose = 0)
                best_pred_val = np.amax(predictions)
                best_pred_indx = np.argmax(predictions)

                print ('[PREDICTION]')
                #print ('Predictions: ', predictions)
                print ('Highest prediction value: ', best_pred_val)
                print ('Traffic sign number: ', best_pred_indx)
                print ('Current index: ', (x_ptr,y_ptr))

                if(predictions[0, np.argmax(predictions)] > highest_pred):
                    highest_pred = predictions[0, np.argmax(predictions)]
                    highest_pred_pos = [x_ptr, y_ptr]
                    highest_pred_filter = filter

                x_ptr += strides[0]
            x_ptr = 0
            y_ptr += strides[1]

        #Paint green box around the object found
        np_image[highest_pred_pos[1]:highest_pred_pos[1]+highest_pred_filter:1,
            highest_pred_pos[0]:highest_pred_filter + highest_pred_pos[0]:1] = [0,255,0]
        print('[RESULTS!]')
        print('Highest pred (x_ptr,y_ptr): ', (highest_pred_pos[0], highest_pred_pos[1]))
        print('Highest pred value', highest_pred)
        print('Highest pred filter: ', highest_pred_filter)
        plt.imshow(np_image)
        plt.show()

images = load_traffic_signs.load_traffic_scenes('traffic_scenes/FullIJCNN2013/', [40,3])
print('[INFO] Looking at picture 40...')
image = images[0]
plt.imshow(image)
plt.show()

custom_cnn_model = load_model('../traffic_sign_recognition/neural_network/saved_models/custom_model.h5')
scan_image(np_image = images[0], strides = (10,10), convnet = custom_cnn_model)