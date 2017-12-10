# Written by Faramarz Munshi and Saeed Karimi
# Copyright Pepperdata 2017

# ************************************************ Import Libraries ****************************************************

import numpy as np
import random
import matplotlib.pyplot as plt
from pprint import pprint
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.utils import np_utils
import numpy as np
import pickle
import keras
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from collections import Counter
import os



NUM_EPOCHS = 5000

# **************************************************** Functions *******************************************************
def _save_obj(obj, name):
    """
    Saves an object in a pickle file.
    :param obj: object to be saved
    :param name: name of file/file path
    """
    directory = '/bigdisk/pickles/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(directory + name, 'w') as outfile:
        pickle.dump(obj, outfile)

def _split_data_fairly(x_train, y_train, x_test, y_test):
    """
    Split data into training and test data with certain ratio from each label
    :param x_whole: A matrix in which rows correspond to feature vectors
    :param y_whole: A vector containing labels
    :return: training and test data/label
    """
    print "Splitting data fairly..."
    # Concatenate the train and test datasets
    x_whole = np.concatenate((x_train,x_test), axis=0)
    y_whole = np.concatenate((y_train,y_test), axis=0)
 
    MIN_NUM_PER_LABEL = 0
    # Ratio between number of training and test data
    ratio = 0.20

    # All possible labels
    all_labels = np.unique(y_whole)
    labels_to_remove = []
    # Calculate the number of data available for each label
    data_per_label = {}
    for label in all_labels:
        data_per_label[label] = np.sum(y_whole == label)
        if data_per_label[label] < MIN_NUM_PER_LABEL:
            labels_to_remove.append(label)

   # Initialization
    training_data = np.zeros(shape=(1,x_whole.shape[1]))
    training_label = []
    test_data = np.zeros(shape=(1,x_whole.shape[1]))
    test_label = []

    # Remove rows with less than "MIN_NUM_PER_LABEL"
    for label in labels_to_remove:
        rows = np.where(y_whole == label)[0]
        y_whole = np.delete(y_whole, rows, axis=0)
        x_whole = np.delete(x_whole, rows, axis=0)
    if labels_to_remove:
        new_y = []
        unique_labels = sorted(np.unique(y_whole))
        for l in y_whole:
            ind = unique_labels.index(l)
            new_y.append(ind)
        y_whole = np.array(new_y)

    print "Removed labels less than", MIN_NUM_PER_LABEL

    # Constitute the training and test data set using random indices 
    # but keeping the ratio between labels the same in test and train set
    for label in all_labels:
        rows = np.where(y_whole == label)[0]
        random.shuffle(rows)
        num_test = int(len(rows) * ratio)

        test_data = np.concatenate((test_data,x_whole[rows[0:num_test],:]), axis=0)
        for i in y_whole[rows[0:num_test]]:
            test_label.append(i)

        training_data = np.concatenate((training_data,x_whole[rows[num_test:],:]), axis=0)
        for i in y_whole[rows[num_test:]]:
            training_label.append(i)
   
    training_data = training_data[1:,:]
    training_label = np.array(training_label)
    test_data = test_data[1:,:]
    test_label = np.array(test_label)

    # Another random shuffling
    indexes = range(training_data.shape[0])
    random.shuffle(indexes)
    training_data = training_data[indexes,:]
    training_label = training_label[indexes]

    indexes = range(test_data.shape[0])
    random.shuffle(indexes)
    test_data = test_data[indexes,:]
    test_label = test_label[indexes]
    
    print "Split data."

    return training_data, training_label, test_data, test_label

def _read_pepper_cnn(fname):
    """
    Reads the output of the cnn_features.py and formats it into a manageable
    format
    :param fname: file name / realm name @vtype: str
    """
    print "Reading and formatting data for ", fname
    x_train = np.array(_load_obj(fname + '^cnn_training_output'))[:,0,:]
    x_test = np.array(_load_obj(fname + '^cnn_test_output'))[:,0,:]
    y_train = np.array(_load_obj(fname + '^training_labels'))
    y_test = np.array(_load_obj(fname + '^test_labels'))
    return x_train, y_train, x_test, y_test

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


def _optimization(dataset1, dataset2, FILEPATH, nb_epochs=3000):
    """
    First trains a model on dataset1, then predicts the labels for vectors in dataset2 using labels of dataset1
    :param dataset1: A dictionary of certain format
    :param dataset2: A dictionary of certain format
    :return: Predicted labels for dataset2 using labels of dataset1
    """
    
    # Decompose into train and test sets for both datasets
    x_train1 = dataset1['train']
    y_train1 = dataset1['labels']['train']
    Y_train1 = dataset1['hot_labels']['train']

    x_test1 = dataset1['test']
    y_test1 = dataset1['labels']['test']
    Y_test1 = dataset1['hot_labels']['test']

    x_train2 = dataset2['train']

    x_test2 = dataset2['test']

    # Batch Size for learner
    batch_size = min(x_train1.shape[0]/7, 16)

    # Number of labels
    num_labels = len( np.unique( np.concatenate((y_train1,y_test1),axis=0) ) )

    # Same description as model defined in main()
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

    early_stopping = EarlyStopping(monitor='loss', min_delta=.01, patience=10, mode='auto')

    model_cp = ModelCheckpoint(FILEPATH, monitor='val_loss', verbose=0, save_best_only=True, mode='auto', period=1)

    # Train on dataset 1
    hist = model1.fit(x_train1, Y_train1, batch_size=batch_size, nb_epoch=nb_epochs, verbose=1,
                      validation_data=(x_test1, Y_test1), shuffle=True, callbacks=[reduce_lr, early_stopping, model_cp])

    dataset2_new_labels_train = []

    # Assign labels to dataset 2 using trained model on dataset 1
    for i in range(x_train2.shape[0]):
        xTrain = x_train2[i,:].reshape((1,x_train2.shape[1]))
        dataset2_new_labels_train.append(np.argmax(model1.predict(xTrain, batch_size=1)))

    dataset2_new_labels_test = []

    for i in range(x_test2.shape[0]):
        xTest = x_test2[i,:].reshape((1,x_test2.shape[1]))
        dataset2_new_labels_test.append(np.argmax(model1.predict(xTest, batch_size=1)))
    
    # Clear the models out of memory so it doesn't mistakenly use them again
    x_model1 = []
    y_model1 = []
    out_model1 = []
    model1 = []

    return dataset2_new_labels_train, dataset2_new_labels_test

def _load_obj(filename):
    """
    Load a pickled file
    :param filename: filename string @vtype = str
    """
    with open('/home/ubuntu/bigdisk/pickles/' + filename) as f:
        return pickle.load(f)

def _to_dictionary(x_train, y_train, x_test, y_test):
    """
    Translate a ucr dataset formatted set of data into a dictionary for merging
    :param x_train: Numpy array with training data      @vtype: np.array
    :param x_test: Numpy array with test data           @vtype: np.array
    :param y_train: Numpy array with training labels    @vtype: np.array
    :param y_test: Numpy array with test labels         @vtype: np.array
    """
    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
    
    # One hot encode label vectors
    Y_train = _one_hot_encode(y_train, nb_classes)
    Y_test = _one_hot_encode(y_test, nb_classes)

    # Create the final dictionary
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
    print "Merging datasets..."
    _THRESHOLD1 = 0

    _THRESHOLD2 = 0

    # Number of labels in dataset 1
    _NUM_LABELS_D1 = len(np.unique(np.concatenate((dataset1['labels']['train'],dataset1['labels']['test']),axis=0)))

    # Number of labels in dataset 2
    _NUM_LABELS_D2 = len(np.unique(np.concatenate((dataset2['labels']['train'],dataset2['labels']['test']),axis=0)))

    print "Optimizing dataset 1 model and running dataset 2 through..."
    FILEPATH = '../' + DATASET_NAMES[0] + '^' + DATASET_NAMES[1]
    # Call the optimization function to train on the first dataset and predict on the second dataset
    ds2_labels_using_ds1_train, ds2_labels_using_ds1_test = _optimization(dataset1, dataset2, FILEPATH, nb_epochs=NUM_EPOCHS)

    # Initialize the label counting matrix
    label_counter = np.zeros(shape=(_NUM_LABELS_D2, _NUM_LABELS_D1))

    # Fill the label counting matrix accordingly
    for i in range(len(ds2_labels_using_ds1_train)):
        label_counter[int(dataset2['labels']['train'][i]), int(ds2_labels_using_ds1_train[i])] += 1

    for i in range(len(ds2_labels_using_ds1_test)):
        label_counter[int(dataset2['labels']['test'][i]), int(ds2_labels_using_ds1_test[i])] += 1

    Matrix1 = np.matrix.copy(label_counter)

    # Initialize the new set of labels for dataset 2
    ds2_new_labels_train = np.zeros(shape=(len(ds2_labels_using_ds1_train), 2))
    ds2_new_labels_test = np.zeros(shape=(len(ds2_labels_using_ds1_test), 2))

    # Determine the new labels for dataset 2
    for i in range(len(ds2_labels_using_ds1_train)):
        true_label = int(dataset2['labels']['train'][i])
        predicted_label = int(ds2_labels_using_ds1_train[i])
        most_frequent_label = np.argmax(label_counter[:, predicted_label])
        if true_label == most_frequent_label:
            ds2_new_labels_train[i, :] = np.array([predicted_label, true_label])
        elif label_counter[true_label, predicted_label] < _THRESHOLD1 * np.sum(label_counter,axis=0)[predicted_label]:
            ds2_new_labels_train[i, :] = np.array([predicted_label, most_frequent_label])
        else:
            ds2_new_labels_train[i, :] = np.array([predicted_label, -1])


    # Determine the new labels for dataset 2
    for i in range(len(ds2_labels_using_ds1_test)):
        true_label = int(dataset2['labels']['test'][i])
        predicted_label = int(ds2_labels_using_ds1_test[i])
        most_frequent_label = np.argmax(label_counter[:, predicted_label])
        if true_label == most_frequent_label:
            ds2_new_labels_test[i, :] = np.array([predicted_label, true_label])
        elif label_counter[true_label, predicted_label] < _THRESHOLD1 * np.sum(label_counter,axis=0)[predicted_label]:
            ds2_new_labels_test[i, :] = np.array([predicted_label, most_frequent_label])
        else:
            ds2_new_labels_test[i, :] = np.array([predicted_label, -1])

    print "Optimizing dataset 2 model and running dataset 1 through..."
    FILEPATH = "../" + DATASET_NAMES[1] + '^' + DATASET_NAMES[0]
    # Call the optimization function to train on the second dataset and predict on the first dataset
    ds1_labels_using_ds2_train, ds1_labels_using_ds2_test = _optimization(dataset2, dataset1, FILEPATH, nb_epochs=NUM_EPOCHS)

    # Initialize the label counting matrix
    label_counter = np.zeros(shape=(_NUM_LABELS_D1, _NUM_LABELS_D2))

    # Fill the label counting matrix accordingly
    for i in range(len(ds1_labels_using_ds2_train)):
        label_counter[int(dataset1['labels']['train'][i]), int(ds1_labels_using_ds2_train[i])] += 1

    for i in range(len(ds1_labels_using_ds2_test)):
        label_counter[int(dataset1['labels']['test'][i]), int(ds1_labels_using_ds2_test[i])] += 1

    Matrix2 = np.matrix.copy(label_counter.T)

    # Initialize the new set of labels for dataset 1
    ds1_new_labels_train = np.zeros(shape=(len(ds1_labels_using_ds2_train), 2))
    ds1_new_labels_test = np.zeros(shape=(len(ds1_labels_using_ds2_test), 2))

    # Determine the new labels for dataset 1
    for i in range(len(ds1_labels_using_ds2_train)):
        true_label = int(dataset1['labels']['train'][i])
        predicted_label = int(ds1_labels_using_ds2_train[i])
        most_frequent_label = np.argmax(label_counter[true_label, :])
        if predicted_label == most_frequent_label:
            ds1_new_labels_train[i, :] = np.array([true_label, predicted_label])
        elif label_counter[true_label, predicted_label] < _THRESHOLD2 * np.sum(label_counter, axis=1)[true_label]:
            ds1_new_labels_train[i, :] = np.array([true_label, most_frequent_label])
        else:
            ds1_new_labels_train[i, :] = np.array([true_label, -1])

    # Determine the new labels for dataset 1
    for i in range(len(ds1_labels_using_ds2_test)):
        true_label = int(dataset1['labels']['test'][i])
        predicted_label = int(ds1_labels_using_ds2_test[i])
        most_frequent_label = np.argmax(label_counter[true_label, :])
        if predicted_label == most_frequent_label:
            ds1_new_labels_test[i, :] = np.array([true_label, predicted_label])
        elif label_counter[true_label, predicted_label] < _THRESHOLD2 * np.sum(label_counter, axis=1)[true_label]:
            ds1_new_labels_test[i, :] = np.array([true_label, most_frequent_label])
        else:
            ds1_new_labels_test[i, :] = np.array([true_label, -1])

    print "Composing new dataset..."
    # Concatenate all labels from both datasets

    all_labels = np.concatenate((ds1_new_labels_train, ds2_new_labels_train,
                                 ds1_new_labels_test, ds2_new_labels_test), axis=0)

    # Transform the tuple labels to scalar labels
    already_explored_rows = []

    label = 0

    vector_label = np.zeros(shape=(all_labels.shape[0], 1))

    for i in range(all_labels.shape[0]):
        if np.where((all_labels == all_labels[i, :]).all(axis=1))[0][0] not in already_explored_rows:
            rows = np.where((all_labels == all_labels[i, :]).all(axis=1))[0]
            vector_label[rows] = label
            label += 1
            for j in range(len(rows)):
                already_explored_rows.append(rows[j])

    vector_label = np.squeeze(vector_label)

    # One hot encoded version of the labels
    hot_labels = _one_hot_encode(vector_label, len(set(vector_label)))

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

    print "Finished Merging."
    # Return the merged dataset as a dictionary
    return new_dataset, Matrix1, Matrix2

# ************************************************* Merge two Datasets *************************************************

def main():
    # List of realms whose data you want to use to train the model
    #rlist = ['dataminesjc', 'rubicon-fmap', 'gamut-prod', 'rubicon-fbmp', 'nbcuni-superstore', 'jumpshot-jsc', 'rubicon-fbmq', 'rally-health-integration', 'prudential-nj-exp2', 'rubicon-fmaq']
    rlist = ['jumpshot-jsc']    
    # Make sure that the realms have been through "data_load_and_format.py" and then "cnn_features.py"
    rlist = rlist[::-1]
    # Reads the datasets and splits them into train and test fairly then translates into dictionary format
    x_train, y_train, x_test, y_test = _read_pepper_cnn(rlist[0])
    x_train, y_train, x_test, y_test = _split_data_fairly(x_train, y_train, x_test, y_test)
    dataset = _to_dictionary(x_train, y_train, x_test, y_test)

    # Maximum number of epochs for the code to run, although early stopping is implemented

    # Iterate through all realms and merge the datasets
    for num, realm in enumerate(rlist[1:]):
        # Read, split, and to dictionary format
        x_train, y_train, x_test, y_test = _read_pepper_cnn(realm)
        x_train, y_train, x_test, y_test = _split_data_fairly(x_train, y_train, x_test, y_test)  
        dataset1 = _to_dictionary(x_train, y_train, x_test, y_test)
        
        # Current realm it's merging, along with all relevant details about the realm
        print "Current Merging Realm: {0}".format(realm)
        print "Dimensions of training data and test data: {0}, {1}".format(x_train.shape, x_test.shape)
        nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
        print "Number of labels: {0}".format(nb_classes)
        countertrain = dict(Counter(y_train))
        countertest = dict(Counter(y_test))
        print "data per class train: ", countertrain
        print "data per class test: ", countertest

        print "Combining ", rlist[1:num+1], " with ", realm
        names = rlist[0]
        for r in rlist[1:num+1]:
            names += '_' + r
        # Global to keep track of which have been merged and which haven't
        global DATASET_NAMES
        DATASET_NAMES = (names, realm)
        # Merge the datasets
        dataset, matrix1, matrix2 = _merge_datasets(dataset,dataset1)

    # Pull out data from dataset model after all merging is complete
    x_train = dataset['train']
    x_test = dataset['test']
    y_train = dataset['labels']['train']
    y_test = dataset['labels']['test']
    Y_train = dataset['hot_labels']['train']
    Y_test = dataset['hot_labels']['test']

    # Number of time series with specific label for training data
    label_index_train = {}

    for i in range(len(np.unique(y_train))):
        label_index_train[i] = np.sum(y_train == i)

    # Number of time series with specific label for test data
    label_index_test = {}

    for i in range(len(np.unique(y_test))):
        label_index_test[i] = np.sum(y_test == i)

    print 'Number of time series per label in training data is = {0}'.format(label_index_train)

    print 'Number of time series per label in test data is = {0}'.format(label_index_test)

    print 'Number of time series in training data is = {0}'.format(x_train.shape[0])

    print 'Number of time series in test data is = {0}'.format(x_test.shape[0])

    # Number of labels
    num_labels = int(max(max(y_train),max(y_test))+1)

    # either 16 or the number of time series over 7, which ever is smaller
    batch_size = min(x_train.shape[0]/7, 16)

    # The model design (Input vector, into a dropout of 10% into a dense layer of 50 nodes
    # with activation function relu into another dropout of 20% into another dense layer of
    # 50 hidden nodes again with activation function relu, into a softmax into the output layer
    x_model = Input(x_train.shape[1:])
    y_model = Dropout(0.1)(x_model)
    y_model = Dense(50, activation='relu')(x_model)
    y_model = Dropout(0.2)(y_model)
    y_model = Dense(50, activation='relu')(y_model)
    out_model = Dense(num_labels, activation='softmax')(y_model)

    # Define our model
    model = Model(input=x_model, output=out_model)

    # Define our optimizer and compile with the loss function
    optimizer = keras.optimizers.Adadelta()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Callbacks: reduce_lr, early_stopping, and model_cp (reduces learning rate on the final curve and 
    # stops the training early if the loss doesn't decrease in the next epoch) Model_cp saves the best model into the file "FINAL".
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=200, min_lr=0.1)

    early_stopping = EarlyStopping(monitor='loss', min_delta=.01, patience=5, mode='auto')

    model_cp = ModelCheckpoint('../GAMUT', monitor='val_loss', verbose=0, save_best_only=True, mode='auto', period=1)

    hist = model.fit(x_train, Y_train, batch_size=batch_size, nb_epoch=NUM_EPOCHS, verbose=1,
                     validation_data=(x_test,Y_test), shuffle=True, callbacks=[reduce_lr, early_stopping, model_cp])

    print 'Number of time series after removing unnecessary labels in training data is = {0}'.format(x_train.shape[0])

    print 'Number of time series after removing unnecessary labels in test data is = {0}'.format(x_test.shape[0])

    print 'Number of labels is = {0}'.format(num_labels)

    print 'Predicted labels vs True labels'

    y_pred = []
    # List of predictions on the test set
    for i in range(x_test.shape[0]):
        xTest = x_test[i,:].reshape((1,2048))
        y_pred.append(int(np.argmax(model.predict(xTest, batch_size=1))))

    # Calculate the accuracy per label
    accuracy_per_label = {}
    y_pred = np.array(y_pred)
    y_test = np.array(y_test)
    for i in range(num_labels):
        pred_label_loc = np.where(y_pred==i)[0]
        actual_label_loc = np.where(y_test==i)[0]
        accuracy_per_label[i] = 0
        for j in pred_label_loc:
            if j in actual_label_loc:
                accuracy_per_label[i] += 1
        if actual_label_loc.tolist():
            accuracy_per_label[i] = (accuracy_per_label[i]*100.0/len(actual_label_loc), len(actual_label_loc))

    # If the number of labels is 2, calculate the precision, recall and F1beta score
    y_pred = y_pred.tolist()
    if num_labels == 2:
        y_test = y_test.tolist()
        print 'Precision was {0}.'.format(precision(y_test,y_pred))
        print 'Recall was {0}.'.format(recall(y_test,y_pred))
        print 'F1 was {0}.'.format(fbeta_score(y_test,y_pred))

    # Print the accuracy per label
    print 'Accuracy per label:'
    pprint(accuracy_per_label)

    print rlist
    y_whole = np.concatenate((y_train, y_test), axis=0)
    _save_obj(y_whole, 'final_label_vector')

if __name__ == '__main__':
    main()
