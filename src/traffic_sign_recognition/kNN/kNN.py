from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import cv2
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import color, exposure



def image_to_feature_vector(image, size =(32,32)):
    ''' Resize the image to a fixed size (32x32), then flatten the image into
        a list of raw pixel intensities
 
    :param image: image 
    :return: single list of 32x32x3 numbers (flatten RGB pixel intensities)
    '''
    return cv2.resize(image, size).flatten()


def extract_color_histogram(image, bins=(8,8,8)):

    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)

    # return the flattened histogram as the feature vector
    return hist.flatten()


def extract_hog_features(image):

    image = color.rgb2grey(cv2.resize(image, (32,32)))
    feature = hog(image,orientations=9, pixels_per_cell=(16, 16),
                  cells_per_block=(1, 1),visualise=False)

    return feature

def classify(training_features, training_labels, test_features, test_labels, k ):
    '''  '''
    # show information on the memory consumed by the feature sets:
    training = np.array(training_features)
    test = np.array(test_features)
    print("[INFO] training_feature array consumes: {:.2f}MB".format(
        training.nbytes / (1024 * 1000.0)),"of memory.")
    print("[INFO] test_feature array consumes: {:.2f}MB".format(
        test.nbytes / (1024 * 1000.0)), "of memory.")


    print("[INFO] evaluating accuracy...")

    # train k-NN classifier
    model = KNeighborsClassifier(n_neighbors=k,
                                 n_jobs= -1)
    model.fit(training_features, training_labels)

    #evaluate k-NN classifier
    acc = model.score(test_features, test_labels)
    print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))


def kNN_classifier(images_training, labels_training, images_test, labels_test ):
    '''
    
    :return: 
    '''

    # initialize the raw pixel intensities matrices, the features matrices, the hog matrices
    # and labels list
    raw_im_training = []
    features_im_training = []
    hog_im_training = []
    raw_im_test = []
    features_im_test = []
    hog_im_test = []

    # loop over the input images
    print("[INFO] extracting feature vectors and histogram from training set...")
    for image in images_training:
        # extract raw pixel intensity "features", followed by a color
        # histogram to characterize the color distribution of the pixels
        # in the image
        pixels = image_to_feature_vector(image)
        hist = extract_color_histogram(image)
        hog = extract_hog_features(image)

        # update the raw images, features, and labels matrices,
        # respectively
        raw_im_training.append(pixels)
        features_im_training.append(hist)
        hog_im_training.append(hog)

    print("[INFO] extracting feature vectors and histogram from test set...")
    for image in images_test:
        # extract raw pixel intensity "features", followed by a color
        # histogram to characterize the color distribution of the pixels
        # in the image
        pixels = image_to_feature_vector(image)
        hist = extract_color_histogram(image)
        hog = extract_hog_features(image)
        # update the raw images, features, and labels matrices,
        # respectively
        raw_im_test.append(pixels)
        features_im_test.append(hist)
        hog_im_test.append(hog)

    #show some information on the memory consumed by the raw images
    # matrix and features matrix
    raw_im_training = np.array(raw_im_training)
    raw_im_test = np.array(raw_im_test)
    #labels = np.array(labels)
    print("[INFO] raw pixels training set matrix: {:.2f}MB".format(
        raw_im_training.nbytes / (1024 * 1000.0)))
    print("[INFO] raw pixel test set matrix: {:.2f}MB".format(
       raw_im_test.nbytes / (1024 * 1000.0)))
    #print("[INFO] color histogram training set matrix: {:.2f}MB".format(
    #    raw_im_test.nbytes / (1024 * 1000.0)))
    #print("[INFO] color histogram test set matrix: {:.2f}MB".format(
    #    raw_im_test.nbytes / (1024 * 1000.0)))
    #print("[INFO] raw pixel test set matrix: {:.2f}MB".format(
    #    raw_im_test.nbytes / (1024 * 1000.0)))
    #print("[INFO] raw pixel test set matrix: {:.2f}MB".format(
    #    raw_im_test.nbytes / (1024 * 1000.0)))

    # train and evaluate a k-NN classifer on the raw pixel intensities
    for k in range (1,3):
        print("[INFO:] k = ",k)
        print("[INFO] evaluating accuracy...")
        model = KNeighborsClassifier(n_neighbors=k,
                                     n_jobs=-1)
        model.fit(raw_im_training, labels_training)
        acc = model.score(raw_im_test, labels_test)
        print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

        # train and evaluate a k-NN classifier on the histogram
        # representations
        print("[INFO] evaluating histogram accuracy...")
        model = KNeighborsClassifier(n_neighbors=k,
                                     n_jobs=-1)
        model.fit(features_im_training, labels_training)
        acc = model.score(features_im_test, labels_test)
        print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))

        # train and evaluate a k-NN classifier on the hog
        # representations
        print("[INFO] evaluating histogram accuracy...")
        model = KNeighborsClassifier(n_neighbors=k,
                                    n_jobs=-1)
        model.fit(hog_im_training, labels_training)
        acc = model.score(hog_im_test, labels_test)
        print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))
