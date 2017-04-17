import csv
import matplotlib.pyplot as plt

# This module only works for the specific case of reading traffic signs and is not
# intended as a general file reading module.


def training(rootpath):
    images = []
    labels = []
    for c in range(0,1):
        prefix = rootpath + '/' + format(c, '05d') + '/'
        gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')
        gtReader = csv.reader(gtFile, delimiter=';')
        gtReader.__next__()
        for row in gtReader:
            images.append(plt.imread(prefix + row[0]))
            labels.append(row[7])
        gtFile.close()
    return images, labels

def testing(rootpath):
    images = []
    labels = []
    prefix = rootpath + '/'
    gtFile = open(prefix + 'GT-final_test' + '.csv')
    gtReader = csv.reader(gtFile, delimiter=';')
    gtReader.__next__()
    for row in gtReader:
        images.append(plt.imread(prefix + row[0]))
        labels.append(row[7])
    gtFile.close()
    return images, labels