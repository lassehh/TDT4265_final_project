import numpy as np
import os
import skimage.transform
import cv2
import csv
#import matplotlib.cm
#import time
import matplotlib.pyplot as plt

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels

def load_data(rootpath):
    ''' Reads traffic sign data for German Traffic Sign Recognition Benchmark.
    Arguments: path to the traffic sign data, for example "./GTSRB/Training"
    Returns:   list of images, list of corresponding labels '''

    images = []  # images
    labels = []  # corresponding labels

    # loop over the first two classes, to loop over all 42 classes replace 3 with 43
    for c in range(0, 3):
        prefix = rootpath + '/' + format(c, '05d') + '/'  # subdirectory for class
        gt_file = open(prefix + 'GT-' + format(c, '05d') + '.csv')  # annotations file
        gt_reader = csv.reader(gt_file, delimiter=';')  # csv parser for annotations file
        gt_reader.__next__() # skip header
        # loop over all images in current annotations file
        for row in gt_reader:
            image = plt.imread(prefix + row[0])  # the 1th column is the filename
            labels.append(row[7])  # the 8th column is the label
            # extract the region of interest from image
            x1 = np.int(row[3])
            y1 = np.int(row[4])
            x2 = np.int(row[5])
            y2 = np.int(row[6])
            roi_image = image[y1:y2,x1:x2,:]
            images.append(roi_image)
        gt_file.close()

    return images, labels

def extract_features(images,labels):
    """This function implements hog detection (histogram of oriented gradient). Used hog because it was one of 
    the data sets from the web page ready to download. 
    
    :param images: region of interest images 
    :param labels: label in index corresponding to picture in index 
    :return data: The data that came out, have to read a little bit more about this.
    """
    block_size = (32 // 2, 32 // 2)
    block_stride = (32 // 4, 32 // 4)
    cell_size = block_stride
    nbins = 9
    hog = cv2.HOGDescriptor((32,32), block_size, block_stride,cell_size, nbins) #http://docs.opencv.org/2.4/modules/gpu/doc/object_detection.html
    data = [hog.compute(x.astype(np.uint8)) for x in images]
    return data


def resize_images(images):
    images32 = [skimage.transform.resize(image, (48, 48))
     for image in images]
    return images32

def main():
    path = os.path.abspath('.')
    rootpath = os.path.join(path, 'GTSRB/Final_Training/Images') # When this file is in the same directory as the GTSRB folder
    im,la = load_data(rootpath)
    im32 = resize_images(im)
    data = extract_features(im32,la)



if __name__ == '__main__': main()