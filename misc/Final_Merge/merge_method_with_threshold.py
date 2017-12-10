# ************************************************ Import Libraries ****************************************************

import numpy as np
import random

np.random.seed(813306)

from pprint import pprint
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.utils import np_utils
import numpy as np
import pickle
import keras
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from sklearn.decomposition import PCA
import pandas as pd

_VERBOSE = 1
NUM_EPOCHS = 2
# **************************************************** Functions *******************************************************
def _interpolation(matrix):
    """
    Interpolate NAN elements in matrix.
    :param matrix: input matrix
    :return: @rtype = np.array, interpolated matrix
    """
    try:
        ok = ~np.isnan(matrix)
        xp = ok.ravel().nonzero()[0]
        fp = matrix[~np.isnan(matrix)]
        x = np.isnan(matrix).ravel().nonzero()[0]
        matrix[np.isnan(matrix)] = np.interp(x, xp, fp)
        return matrix
    except:
        return matrix

def read_format_pepper():
    _METRICS = ['vmsram', 'tasks', 't_rscthnetno', 't_rscthhfsrb', 'c_ucpupct']
    train_data = _load_obj('training_data')
    test_data = _load_obj('test_data')
    print("Loaded objects.")
    train_labels = train_data['labels']
    test_labels = test_data['labels']
    print("Formatting Pepperdata test data...")
    test_data = _interpolation(np.concatenate((test_data[_METRICS[0]],test_data[_METRICS[1]],test_data[_METRICS[2]], test_data[_METRICS[3]], test_data[_METRICS[4]]), axis=1))
    print("Formatting Pepperdata train data...")
    train_data = _interpolation(np.concatenate((train_data[_METRICS[0]],train_data[_METRICS[1]],train_data[_METRICS[2]], train_data[_METRICS[3]], train_data[_METRICS[4]]), axis=1))
    print("Done formatting.")
    return train_data, train_labels, test_data, test_labels

def precision(y_true, y_pred):
    """Precision metric.
-
-    Only computes a batch-wise average of precision.
-
-    Computes the precision, a metric for multi-label classification of
-    how many selected items are relevant.
-    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
 -
 -    Only computes a batch-wise average of recall.
 -
 -    Computes the recall, a metric for multi-label classification of
 -    how many relevant items are selected.
 -    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
 -
 -    The F score is the weighted harmonic mean of precision and recall.
 -    Here it is only computed as a batch-wise average, not globally.
 -
 -    This is useful for multi-label classification, where input samples can be
 -    classified as sets of labels. By only using accuracy (precision) a model
 -    would achieve a perfect score by simply assigning every class to every
 -    input. In order to avoid this, a metric should penalize incorrect class
 -    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
 -    computes this, as a weighted mean of the proportion of correct class
 -    assignments vs. the proportion of incorrect class assignments.
 -
 -    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
 -    correct classes becomes more important, and with beta > 1 the metric is
 -    instead weighted towards penalizing incorrect class assignments.
 -    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def _one_hot_encode(label_vector, total_num_labels):
    """
    One hot encode a label vector.
    :param label_vector: a vector of labels for each time series
    :param total_num_labels: total number of labels
    :return: one hot encoded version of labels of all time series
    """
    out = np.zeros(shape=(len(label_vector), total_num_labels))
    for i in range(len(label_vector)):
        out[i, int(label_vector[i])] = 1
    return out


def _optimization(dataset1, dataset2, nb_epochs=3000):
    """
    First trains a model on dataset1, then predicts the labels for vectors in dataset2 using labels of dataset1
    :param dataset1: A dictionary of certain format
    :param dataset2: A dictionary of certain format
    :return: Predicted labels for dataset2 using labels of dataset1
    """
    print "Optimizing..."

    x_train1 = dataset1['train']
    y_train1 = dataset1['labels']['train']
    Y_train1 = dataset1['hot_labels']['train']

    x_test1 = dataset1['test']
    y_test1 = dataset1['labels']['test']
    Y_test1 = dataset1['hot_labels']['test']

    batch_size = min(x_train1.shape[0]/7, 16)

    num_labels = len(np.unique(y_test1))

    x_train2 = dataset2['train']
    x_test2 = dataset2['test']

    x_model1 = Input(x_train1.shape[1:])
    y_model1 = Dropout(0.1)(x_model1)
    y_model1 = Dense(50, activation='relu')(x_model1)
    y_model1 = Dropout(0.2)(y_model1)
    y_model1 = Dense(50, activation='relu')(y_model1)
    out_model1 = Dense(num_labels, activation='softmax')(y_model1)

    model1 = Model(input=x_model1, output=out_model1)

    optimizer = keras.optimizers.Adadelta()
    model1.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=200, min_lr=0.1)

    hist = model1.fit(x_train1, Y_train1, batch_size=batch_size, epochs=nb_epochs, verbose=_VERBOSE,
                      validation_data=(x_test1, Y_test1), shuffle=True, callbacks=[reduce_lr])

    log = pd.DataFrame(hist.history)
    max_test_accuracy = max(log.val_acc.values)
    dataset1_test_error = 1-max_test_accuracy

    print "Max Test Accuracy: {0}".format(max_test_accuracy)

    dataset2_new_labels_train = []

    for i in range(x_train2.shape[0]):
        xTrain = x_train2[i,:].reshape((1,x_train2.shape[1]))
        dataset2_new_labels_train.append(np.argmax(model1.predict(xTrain, batch_size=1)))

    dataset2_new_labels_test = []

    for i in range(x_test2.shape[0]):
        xTest = x_test2[i,:].reshape((1,x_test2.shape[1]))
        dataset2_new_labels_test.append(np.argmax(model1.predict(xTest, batch_size=1)))

    x_model1 = []
    y_model1 = []
    out_model1 = []
    model1 = []

    print "Optimized!"

    return dataset2_new_labels_train, dataset2_new_labels_test, dataset1_test_error


def _load_obj(filename):
    with open(filename) as f:
        return pickle.load(f)

def _readucr(filename):
    try:
       data = np.loadtxt(filename, delimiter = ',')
       Y = data[:,0]
       X = data[:,1:]
    except:
       data = _load_obj(filename)
       data = np.array([[d[0]]+d[1] for d in data])
    Y = data[:,0]
    X = data[:,1:]
    return X, Y


def _ucr_to_dictionary(x_train, y_train, x_test, y_test):
    # x_train, y_train = _readucr('/home/ubuntu/big_disk/GADF_images/' + fname + '_GADF_CNNOUT_' + 'TRAIN')
    #x_test, y_test = _readucr('/home/ubuntu/big_disk/GADF_images/' + fname + '_GADF_CNNOUT_' + 'TEST')
    #x_train, y_train = _readucr(fname + '_GADF_CNNOUT_' + 'TRAIN')
    #x_test, y_test = _readucr(fname + '_GADF_CNNOUT_' + 'TEST')


    nb_classes = len(np.unique(y_test))

    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    dictionary = {'labels':{}, 'hot_labels':{}}

    dictionary['train'] = x_train
    dictionary['test'] = x_test

    dictionary['labels']['train'] = y_train
    dictionary['labels']['test'] = y_test

    dictionary['hot_labels']['train'] = Y_train
    dictionary['hot_labels']['test'] = Y_test

    return dictionary


def _merge_datasets(dataset1, dataset2):
    """
    Merge two datasets to unify their labels
    :param dataset1: A dataset of certain format                                   @vtype: A dictionary
    :param dataset2: A dataset of certain format                                   @vtype: A dictionary
    :return: A merged dataset with unified labels of the same format as inputs     @vtype: A dictionary
    """

#    _THRESHOLD = 0
    print "Merging Datasets..."

    # Number of labels in dataset 1
    _NUM_LABELS_D1 = len(np.unique(dataset1['labels']['test']))

    # Number of labels in dataset 2
    _NUM_LABELS_D2 = len(np.unique(dataset2['labels']['test']))

    # Call the optimization function to train on the first dataset and predict on the second dataset
    ds2_labels_using_ds1_train, ds2_labels_using_ds1_test, _ = \
        _optimization(dataset1, dataset2, nb_epochs=NUM_EPOCHS)

    # Initialize the label counting matrix
    label_counter = np.zeros(shape=(_NUM_LABELS_D2, _NUM_LABELS_D1))

    # Fill the label counting matrix accordingly
    for i in range(len(ds2_labels_using_ds1_train)):
        label_counter[int(dataset2['labels']['train'][i]), int(ds2_labels_using_ds1_train[i])] += 1

    for i in range(len(ds2_labels_using_ds1_test)):
        label_counter[int(dataset2['labels']['test'][i]), int(ds2_labels_using_ds1_test[i])] += 1

    matrix1 = np.matrix.copy(label_counter)

    # Initialize the new set of labels for dataset 2
    ds2_new_labels_train = np.zeros(shape=(len(ds2_labels_using_ds1_train), 2))
    ds2_new_labels_test = np.zeros(shape=(len(ds2_labels_using_ds1_test), 2))

    # Determine the new labels for dataset 2 train
    for i in range(len(ds2_labels_using_ds1_train)):
        true_label = int(dataset2['labels']['train'][i])
        predicted_label = int(ds2_labels_using_ds1_train[i])
        most_frequent_true_label = np.argmax(label_counter[:, predicted_label])
        if true_label == most_frequent_true_label:
            new_label = np.array([predicted_label, true_label])
#        elif label_counter[true_label , predicted_label] < \
  #                      _THRESHOLD * np.sum(label_counter, axis=0)[predicted_label]:
 #           new_label = np.array([predicted_label, most_frequent_true_label])
        else:
            new_label = np.array([predicted_label, -1])

        ds2_new_labels_train[i, :] = new_label

    for i in range(len(ds2_labels_using_ds1_test)):
        true_label = int(dataset2['labels']['test'][i])
        predicted_label = int(ds2_labels_using_ds1_test[i])
        most_frequent_true_label = np.argmax(label_counter[:, predicted_label])
        if true_label == most_frequent_true_label:
            new_label = np.array([predicted_label, true_label])
   #     elif label_counter[true_label, predicted_label] < \
    #                    _THRESHOLD * np.sum(label_counter, axis=0)[predicted_label]:
     #       new_label = np.array([predicted_label, most_frequent_true_label])
        else:
            new_label = np.array([predicted_label, -1])

        ds2_new_labels_test[i, :] = new_label

    print "Created labels for dataset 2 using dataset 1."
    # Call the optimization function to train on the second dataset and predict on the first dataset
    ds1_labels_using_ds2_train, ds1_labels_using_ds2_test, _ = \
        _optimization(dataset2, dataset1, nb_epochs=NUM_EPOCHS)

    # Initialize the label counting matrix
    label_counter = np.zeros(shape=(_NUM_LABELS_D1, _NUM_LABELS_D2))

    # Fill the label counting matrix accordingly
    for i in range(len(ds1_labels_using_ds2_train)):
        label_counter[int(dataset1['labels']['train'][i]), int(ds1_labels_using_ds2_train[i])] += 1

    for i in range(len(ds1_labels_using_ds2_test)):
        label_counter[int(dataset1['labels']['test'][i]), int(ds1_labels_using_ds2_test[i])] += 1

    matrix2 = np.matrix.copy(label_counter.T)

    # Initialize the new set of labels for dataset 1
    ds1_new_labels_train = np.zeros(shape=(len(ds1_labels_using_ds2_train), 2))
    ds1_new_labels_test = np.zeros(shape=(len(ds1_labels_using_ds2_test), 2))

    # Determine the new labels for dataset 1
    for i in range(len(ds1_labels_using_ds2_train)):
        true_label = int(dataset1['labels']['train'][i])
        predicted_label = int(ds1_labels_using_ds2_train[i])
        most_frequent_true_label = np.argmax(label_counter[:, predicted_label])
        if true_label == most_frequent_true_label:
            new_label = np.array([predicted_label, true_label])
  #      elif label_counter[true_label, predicted_label] < \
  #                      _THRESHOLD * np.sum(label_counter, axis=0)[predicted_label]:
  #          new_label = np.array([predicted_label, most_frequent_true_label])
        else:
            new_label = np.array([predicted_label, -1])

        ds1_new_labels_train[i, :] = new_label

    # Determine the new labels for dataset 1
    for i in range(len(ds1_labels_using_ds2_test)):
        true_label = int(dataset1['labels']['test'][i])
        predicted_label = int(ds1_labels_using_ds2_test[i])
        most_frequent_true_label = np.argmax(label_counter[:, predicted_label])
        if true_label == most_frequent_true_label:
            new_label = np.array([predicted_label, true_label])
  #      elif label_counter[true_label, predicted_label] < \
  #                      _THRESHOLD * np.sum(label_counter, axis=0)[predicted_label]:
  #          new_label = np.array([predicted_label, most_frequent_true_label])
        else:
            new_label = np.array([predicted_label, -1])

        ds1_new_labels_test[i, :] = new_label

    print "Created labels for dataset 1 using dataset 2."

    # Concatenate all labels from both datasets

    all_labels = np.concatenate((ds1_new_labels_train, ds2_new_labels_train,
                                 ds1_new_labels_test, ds2_new_labels_test), axis=0)

    unique_labels = [list(x) for x in set(tuple(x.tolist()) for x in all_labels)]
    vector_label = []
    for label_combo in all_labels:
        vector_label.append(unique_labels.index(label_combo.tolist()))

    vector_label = np.array(vector_label)
    num_labels = len(unique_labels)
    # One hot encoded version of the labels
    hot_labels = _one_hot_encode(vector_label, num_labels)

    vector_label_train = vector_label[0:(ds1_new_labels_train.shape[0] + ds2_new_labels_train.shape[0])]

    vector_label_test = vector_label[(ds1_new_labels_train.shape[0] + ds2_new_labels_train.shape[0]):]

    hot_labels_train = hot_labels[0:(ds1_new_labels_train.shape[0] + ds2_new_labels_train.shape[0]),:]

    hot_labels_test = hot_labels[(ds1_new_labels_train.shape[0] + ds2_new_labels_train.shape[0]):,:]

    # Initialize the concatenated dataset
    new_dataset = {'labels': {'train': vector_label_train, 'test': vector_label_test},
                   'hot_labels': {'train': hot_labels_train, 'test': hot_labels_test}}

    # Fill the corresponding keys for the concatenated dataset
    for key in dataset1.keys():
        if (key != 'labels') and (key != 'hot_labels'):
            new_dataset[key] = np.concatenate((dataset1[key], dataset2[key]), axis=0)

    print "Merged."
    # Return the merged dataset as a dictionary
    return new_dataset, matrix1, matrix2

# ************************************************* Merge two Datasets *************************************************
def main():
    # dataset_list = ['Adiac', 'Beef', 'CBF', 'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee', 'Cricket_X', 'Cricket_Y',
    #                 'Cricket_Z', 'DiatomSizeReduction', 'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR', '50words', 'FISH',
    #                 'Gun_Point', 'Haptics', 'InlineSkate', 'ItalyPowerDemand', 'Lighting2', 'Lighting7', 'MALLAT',
    #                 'MedicalImages','MoteStrain', 'NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2', 'OliveOil',
    #                 'OSULeaf','SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves', 'SwedishLeaf', 'Symbols',
    #                 'synthetic_control', 'Trace', 'TwoLeadECG', 'Two_Patterns', 'uWaveGestureLibrary_X',
    #                 'uWaveGestureLibrary_Y','uWaveGestureLibrary_Z', 'wafer', 'WordsSynonyms', 'yoga']

    flist = ['Cricket_X', 'Cricket_Y', 'Cricket_Z']
    # flist = ['SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII']
    # flist = ['SonyAIBORobotSurface', 'Cricket_X', 'Lighting7']
    flist = ['Lighting2','Lighting7']
    # flist = ['FISH', 'Haptics', 'InlineSkate']
    x_train, y_train = _readucr('/home/ubuntu/big_disk/GADF_images/' + flist[0] + '_GADF_CNNOUT_TRAIN')
    x_test, y_test = _readucr('/home/ubuntu/big_disk/GADF_images/' + flist[0] + '_GADF_CNNOUT_TEST')
    # x_train, y_train, x_test, y_test = read_format_pepper()
    dataset = _ucr_to_dictionary(x_train, y_train, x_test, y_test)

    for num_fname in range(1,len(flist)):
        x_train, y_train = _readucr('/home/ubuntu/big_disk/GADF_images/' + flist[num_fname] + '_GADF_CNNOUT_TRAIN')
        x_test, y_test = _readucr('/home/ubuntu/big_disk/GADF_images/' + flist[num_fname] + '_GADF_CNNOUT_TEST')        
        dataset1 = _ucr_to_dictionary(x_train, y_train, x_test, y_test)
        dataset, matrix1, matrix2 = _merge_datasets(dataset,dataset1)


    train_data = dataset['train']
    test_data = dataset['test']
    labels_train = dataset['labels']['train']
    labels_test = dataset['labels']['test']
    hot_labels_train = dataset['hot_labels']['train']
    hot_labels_test = dataset['hot_labels']['test']
    num_labels = int(max(max(labels_train), max(labels_test)) + 1)

    print 'Number of labels = {0}'.format(num_labels)

    assumed_num_labels = len(np.unique(np.concatenate((labels_train, labels_test), axis=0)))

    assert num_labels == assumed_num_labels, "Number of labels mismatch"

    labels_to_remove = []

    if labels_to_remove:
        remaining_rows_train = []
        remaining_rows_test = []

        for row in range(train_data.shape[0]):
            if labels_train[row] not in labels_to_remove:
                remaining_rows_train.append(row)

        train_data = train_data[remaining_rows_train, :]
        labels_train = labels_train[remaining_rows_train]
        hot_labels_train = hot_labels_train[remaining_rows_train, :]

        for row in range(test_data.shape[0]):
            if labels_test[row] not in labels_to_remove:
                remaining_rows_test.append(row)

        test_data = test_data[remaining_rows_test, :]
        labels_test = labels_test[remaining_rows_test]
        hot_labels_test = hot_labels_test[remaining_rows_test, :]

        y_train_copy = np.matrix.copy(labels_train)
        y_test_copy = np.matrix.copy(labels_test)

        for lbl in labels_to_remove:
            for i in range(len(labels_train)):
                if y_train_copy[i] > lbl:
                    labels_train[i] -= 1

            for i in range(len(labels_test)):
                if y_test_copy[i] > lbl:
                    labels_test[i] -= 1

        num_labels = int(max(max(labels_train), max(labels_test)) + 1)

        hot_labels_train = _one_hot_encode(labels_train, num_labels)

        hot_labels_test = _one_hot_encode(labels_test, num_labels)

        print 'Number of labels after removal: ',num_labels

    batch_size = min(train_data.shape[0]/7, 16)

    x_model = Input(train_data.shape[1:])
    Dropout(0.1)(x_model)
    y_model = Dense(50, activation='relu')(x_model)
    Dropout(0.2)(y_model)
    y_model = Dense(50, activation='relu')(y_model)
    out_model = Dense(num_labels, activation='softmax')(y_model)

    model = Model(input=x_model, output=out_model)

    optimizer = keras.optimizers.Adadelta()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=200, min_lr=0.1)

    hist = model.fit(train_data, hot_labels_train, batch_size=batch_size, nb_epoch=NUM_EPOCHS + 100, verbose=1,
                     validation_data=(test_data, hot_labels_test), shuffle=True, callbacks=[reduce_lr])

    print 'Number of time series in training data is = {0}'.format(train_data.shape[0])

    print 'Number of time series in test data is = {0}'.format(test_data.shape[0])

    print 'Matrix 1 = {0}'.format(matrix1)

    print 'Matrix 2 = {0}'.format(matrix2)


    # Number of time series with specific label for training data
    label_index_train = {}

    for i in range(num_labels):
        label_index_train[i] = np.sum(labels_train == i)

    # Number of time series with specific label for test data
    label_index_test = {}

    for i in range(num_labels):
        label_index_test[i] = np.sum(labels_test == i)

    print 'Number of time series per label in training data is = {0}'.format(label_index_train)

    print 'Number of time series per label in test data is = {0}'.format(label_index_test)

    print 'Number of time series in training data is = {0}'.format(train_data.shape[0])

    print 'Number of time series in test data is = {0}'.format(test_data.shape[0])

    print 'Predicted labels vs True labels'

    y_pred = []

    for i in range(test_data.shape[0]):
        xTest = test_data[i, :].reshape((1, 2048))
        print((np.argmax(model.predict(xTest, batch_size=1)), labels_test[i]))
        y_pred.append(int(np.argmax(model.predict(xTest, batch_size=1))))

    accuracy_per_label = {}
    y_pred = np.array(y_pred)
    labels_test = np.array(labels_test)
    for i in range(num_labels):
        pred_label_loc = np.where(y_pred==i)[0]
        actual_label_loc = np.where(labels_test==i)[0]
        accuracy_per_label[i] = 0
        for j in pred_label_loc:
            if j in actual_label_loc:
                accuracy_per_label[i] += 1
        if actual_label_loc.tolist():
            accuracy_per_label[i] = (accuracy_per_label[i]*100.0/len(actual_label_loc), len(actual_label_loc))

    if num_labels == 2:
        labels_test = labels_test.tolist()
        print 'Precision was {0}.'.format(precision(labels_test, y_pred))
        print 'Recall was {0}.'.format(recall(labels_test, y_pred))
        print 'F1 was {0}.'.format(fbeta_score(labels_test, y_pred))

    print 'Accuracy per label:'
    pprint(accuracy_per_label)
    print flist

if __name__ == '__main__':
    main()
