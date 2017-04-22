import kNN
import load_data
import os


def main():
    ''' Main function for tra. kNN
    
    '''
    print("Traffic sign recognition using the GTSRB dataset and k nearest neighbor classifier")
    print("choose features to be extracted by writing 'grey', 'hsv', 'hog' or 'color_histogram'. by writing "
          "anything else the classification will happen with the raw pixels")  #sift
    feature = input('--->')
    print("Thank you, getting right on it.")

    print("[INFO] loading data...")
    rootpath = os.path.abspath('.')  # When this file is in the same directory as the GTSRB folder
    training_features, training_labels, test_features, test_labels =load_data.load_data(rootpath, True, feature)
    kNN.classify(training_features, training_labels, test_features, test_labels, k= 2)
    #im_training, la_training = load_data.load_data(rootpath_training, roi = bool(False))
    #rootpath_test = os.path.join(path, 'GTSRB/Final_Test/Images')
    #im_test, la_test = load_data.load_testdata(rootpath_test, roi = bool(False))
    #print("[INFO] loaded all images")
    #kNN.kNN_classifier(im_training,la_training,im_test,la_test)


if __name__ == '__main__': main()