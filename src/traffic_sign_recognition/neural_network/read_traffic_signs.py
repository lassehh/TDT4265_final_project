import csv
import matplotlib.pyplot as plt

def readTrainingTrafficSigns(rootpath):

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

def readTestingTrafficSigns(rootpath):
    images = []
    labels = []
    prefix = rootpath + '/'
    gtFile = open(prefix + 'GT-final_test' + '.csv')
    gtReader = csv.reader(gtFile, delimiter=';')
    gtReader.__next__()
    for row in gtReader:
        images.append(plt.imread(prefix + row[0]))
        labels.append(row[7])
        break
    gtFile.close()
    return images, labels