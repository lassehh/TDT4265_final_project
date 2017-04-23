import kNN
import load_data
import os
import time


def main():
    ''' Main function for tra. kNN
    
    '''
    print("Traffic sign recognition using the GTSRB dataset and k nearest neighbor classifier")
    print("choose features to be extracted by writing 'grey', 'hsv', 'hog' or 'color_histogram'. by writing "
          "anything else the classification will happen with the raw pixels")  #sift
    feature = input('--->')
    print("How many neighbors, k do you want to use? ")
    k = int(input('--->'))
    print("Thank you, getting right on it.")



    rootpath = os.path.abspath('.')  # When this file is in the same directory as the GTSRB folder
    start_time = time.time()
    training_features, training_labels, test_features, test_labels =load_data.load_data(rootpath, True, feature)
    print('[INFO] time loading data ' + str(time.time() - start_time))

    kNN.classify(training_features, training_labels, test_features, test_labels, k, feature)


if __name__ == '__main__': main()