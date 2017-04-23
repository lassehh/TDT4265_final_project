import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
from skimage import color

def extract_features(images, feature):

    #resize images to a common small size
    images = [cv2.resize(image,(32,32)) for image in images]


    feature_vec = [np.array([]) for image in images]

    if feature == 'grey':
        temp_feature_vec = [color.rgb2grey(image) for image in images]
        temp_feature_vec = [np.array(f).astype(np.float32)/255 for f in new_feature_vec] #normalize
    elif feature == 'hsv':
        new_feature_vec = [color.rgb2hsv(image) for image in images]
        temp_feature_vec = [np.array(f).astype(np.float32) / 255 for f in new_feature_vec]  # normalize
    elif feature == 'hog':
        block_size = (16, 16) #only supported option
        block_stride = (8, 8)   #only supported option
        cell_size = block_stride
        num_bins = 9
        hog = cv2.HOGDescriptor((32,32), block_size, block_stride, cell_size, num_bins)
        temp_feature_vec = [hog.compute(image) for image in images]  # already normalized (L2-norm) for each block
    elif feature == 'color_histogram':
        temp_feature_vec = []
        for image in images:
            hsv= cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, (8,8,8), [0, 180, 0, 256, 0, 256])
            temp_feature_vec.append(cv2.normalize(hist,hist).flatten())
        return temp_feature_vec
    else:
        temp_feature_vec = images

    temp_feature_vec = [f.flatten() for f in temp_feature_vec]
    feature_vec = [np.append(feature_vec[i],temp_feature_vec[i]) for i in range(len(feature_vec))]
    return feature_vec


def load_data(rootpath, roi, feature):
    """ Function for reading the images
    
    
    Arguments: path to the traffic sign data, for example "./GTSRB/Training"
    Returns:   list of images, list of corresponding labels """

    training_images = []  # training images
    training_labels = []  # corresponding labels
    test_images = []      # test images
    test_labels = []      # corresponding labels
    n_train = 0
    n_test = 0
    # loop over all classes
    for c in range(0, 43):  #change to 43
        prefix = rootpath +'/GTSRB/Final_Training/Images/' + format(c, '05d') + '/'  # subdirectory for class
        gt_file = open(prefix + 'GT-' + format(c, '05d') + '.csv')  # annotations file
        gt_reader = csv.reader(gt_file, delimiter=';')  # csv parser for annotations file
        gt_reader.__next__()  # skip header
        # loop over all images in current annotations file
        if roi:
            for row in gt_reader:
                image = plt.imread(prefix + row[0])  # the 1th column is the filename
                training_labels.append(row[7])  # the 8th column is the label
                roi_x1 = np.int(row[3])  # the 4th column is roi upper left x
                roi_y1 = np.int(row[4])  # the 5th column is roi upper left y
                roi_x2 = np.int(row[5])  # the 6th column is roi lower right x
                roi_y2 = np.int(row[6])  # the 7th column is roi lower right y
                roi_image = image[roi_y1:roi_y2,roi_x1:roi_x2,:]  # extract the region of interest from image
                training_images.append(roi_image)
        else:
            for row in gt_reader:
                image = plt.imread(prefix + row[0])  # the 1th column is the filename
                training_labels.append(row[7])  # the 8th column is the label
                training_images.append(image)
    gt_file.close()

    prefix = rootpath + '/GTSRB/Final_Test/Images/'
    gt_file = open(prefix + 'GT-final_test' + '.csv')  # annotations file
    gt_reader = csv.reader(gt_file, delimiter=';')  # csv parser for annotations file
    gt_reader.__next__()  # skip header

    # loop over all images in current annotations file
    if roi:
        for row in gt_reader:
            image = plt.imread(prefix + row[0])  # the 1th column is the filename
            roi_x1 = np.int(row[3])  # the 3rd column is roi upper left x
            roi_y1 = np.int(row[4])  # the 4th column is roi upper left y
            roi_x2 = np.int(row[5])  # the 5th column is roi lower right x
            roi_y2 = np.int(row[6])  # the 5th column is roi lower right y
            roi_image = image[roi_y1:roi_y2, roi_x1:roi_x2, :]  # extract the region of interest from image
            test_images.append(roi_image)
            test_labels.append(row[7])  # the 8th column is the label
    else:
        for row in gt_reader:
            image = plt.imread(prefix + row[0])
            test_labels.append(row[7])  # the 8th column is the label
            test_images.append(image)
    gt_file.close()

    training_features = extract_features(training_images, feature)
    test_features = extract_features(test_images,feature)


    return training_features, training_labels, test_features, test_labels

def extract_hog_features(image):
    ''' This function implements hog feature extraction for one image. The function shows a step by step figure
     so that it is easier to understand.

    :param images: images 
    :return: 
    '''
    # Read image
    image = np.float32(image)

    # Calculate gradient
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)

    # Calculate gradient magnitude and direction in degrees
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hist = [mag,angle]


    return hist

