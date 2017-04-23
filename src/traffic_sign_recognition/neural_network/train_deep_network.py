import numpy as np
# fix the random seed for reproducibility
seed = 1337
np.random.seed(seed)

import convnet_models
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils, plot_model
from keras.models import load_model
from keras import backend as K
K.set_image_dim_ordering('th')


def model_trainer(convnet_models, batch_size = 100, modelname = 'random',
                  epoches = 15, verbose = 2, generator = False):

    # read images
    print('Loading data as numpy array of arrays...')
    X_train = np.load('data_numpy/X_train_data.npy').astype(np.float32)
    X_test = np.load('data_numpy/X_test_data.npy').astype(np.float32)
    Y_train = np.load('data_numpy/Y_train_data.npy')
    Y_test = np.load('data_numpy/Y_test_data.npy')
    Y_test = np_utils.to_categorical(Y_test)
    Y_train = np_utils.to_categorical(Y_train)
    print('Done!')

    # normalize the color channels over the entire data
    #  and center the data cloud
    X_train /= 255
    X_test /= 255
    X_train -= np.mean(X_train, axis = 0)
    X_test -= np.mean(X_test, axis = 0)

    # build the model
    number_of_classes = Y_test.shape[1]
    model = convnet_models(number_of_classes)

    if (generator == True):
        # use a ImageDataGenerator on the same in order to obtain more variety in data from
        # existing data
        training_datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=0.,
            width_shift_range=0.,
            height_shift_range=0.,
            shear_range=0.1,
            zoom_range=0,
            channel_shift_range=0.,
            fill_mode='nearest',
            cval=0.,
            horizontal_flip=False,
            vertical_flip=False,
            rescale=None,
            preprocessing_function=None,)

        testing_datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=0.,
            width_shift_range=0.,
            height_shift_range=0.,
            shear_range=0.,
            zoom_range=0,
            channel_shift_range=0.,
            fill_mode='nearest',
            cval=0.,
            horizontal_flip=False,
            vertical_flip=False,
            rescale=None,
            preprocessing_function=None, )

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        training_datagen.fit(X_train)
        testing_datagen.fit(X_train)

        # fits the model on batches with real-time data augmentation:
        history = model.fit_generator(training_datagen.flow(X_train, Y_train, batch_size=batch_size),
                        steps_per_epoch=len(X_train)/batch_size,
                        epochs=epoches,
                        verbose=verbose,
                        validation_data=(testing_datagen.flow(X_test, Y_test, batch_size=batch_size)),
                        validation_steps = len(X_test)/batch_size)
        # final evaluation of the model
        scores = model.evaluate(X_test, Y_test, verbose=0)
        print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))
    else:
        # fit the model
        history = model.fit(X_train, Y_train,
                        validation_data=(X_test, Y_test),
                        epochs=epoches,
                        batch_size=batch_size,
                        verbose=verbose)
        # final evaluation of the model
        scores = model.evaluate(X_test, Y_test, verbose=0)
        print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))


    # save model for later use and for visualization
    model.save('saved_models/' + modelname + '.h5')
    #plot_model(model, to_file= '/visual_models/' + modelname + '.png') #TODO: Fix graphviz package

    # visualize the model accuracy vs epoches
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(modelname + ' accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig(modelname + '_acc.png')
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(modelname + ' loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig(modelname + '_loss.png')

model_trainer(convnet_models.custom, modelname = 'custom_model',
              epoches = 250, batch_size = 128, verbose = 2, generator = False)
custom_cnn_model = load_model('saved_models/custom_model.h5')