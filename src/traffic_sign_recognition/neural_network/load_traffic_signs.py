import csv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# This module only works for the specific case of reading traffic signs and is not
# intended as a general file reading module.


def training(rootpath):
    images = []
    labels = []
    i = 0
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/'
        gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')
        gtReader = csv.reader(gtFile, delimiter=';')
        gtReader.__next__()
        for row in gtReader:
            images.append(Image.open(prefix + row[0]))
            images[i] = np.array(images[i].resize((90, 90), Image.ANTIALIAS))
            labels.append(row[7])
            i += 1
            print('Training images loaded: ', i)
        gtFile.close()
    return images, labels

def testing(rootpath):
    images = []
    labels = []
    i = 0
    prefix = rootpath + '/'
    gtFile = open(prefix + 'GT-final_test' + '.csv')
    gtReader = csv.reader(gtFile, delimiter=';')
    gtReader.__next__()
    for row in gtReader:
        images.append(Image.open(prefix + row[0]))
        images[i] = np.array(images[i].resize((90, 90), Image.ANTIALIAS))
        labels.append(row[7])
        i += 1
        print('Testing images loaded: ', i)
    gtFile.close()
    return images, labels