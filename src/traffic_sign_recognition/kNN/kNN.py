from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import time

def classify(training_features, training_labels, test_features, test_labels, k , feature):
    '''  '''
    # show information on the memory consumed by the feature sets:
    training = np.array(training_features)
    test = np.array(test_features)
    print("[INFO] training_feature array consumes: {:.2f}MB".format(
        training.nbytes / (1024 * 1000.0)),"of memory.")
    print("[INFO] test_feature array consumes: {:.2f}MB".format(
        test.nbytes / (1024 * 1000.0)), "of memory.")

    # train k-NN classifier
    start_train_time = time.time()
    model = KNeighborsClassifier(n_neighbors=k, n_jobs= -1)
    model.fit(training_features, training_labels)
    print('[INFO]create and fit k-NN classifier with k = ' + str(k) + ' and feature ' + feature + ' takes time: '+
          str(time.time() - start_train_time) )

    print("[INFO] evaluating accuracy...")
    # evaluate k-NN classifier
    start_eval_time = time.time()
    acc = model.score(test_features, test_labels)
    print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))
    print('[INFO] predicting labels and calculating accuracy in test set with k-NN classifier where'
          ' k = ' + str(k) + ' and feature ' + feature + ' takes time: ' + str(time.time() - start_eval_time))

    start_prediction_time = time.time()
    predicted = model.predict(test_features)
    print('[INFO] predicting labels in test set with k-NN classifier where k = ' + str(k) + ' and feature ' +
          feature + ' takes time: ' + str(time.time() - start_prediction_time))

    # confusion matrix
    con_matrix = confusion_matrix(test_labels, predicted)
    np.set_printoptions(precision=2)
    class_names = np.arange(0,43)
    plt.figure()
    plot_confusion_matrix(con_matrix, classes=class_names, title='Confusion matrix, without normalization')
    plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, title = 'Confusion matrix' , cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if (cm[i, j] > thresh) | (cm[i,j]==0) else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')