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

NUM_EPOCHS = 50

# **************************************************** Functions *******************************************************
def _split_data_fairly(x_train, y_train, x_test, y_test):
   """
   Split whole data into training and test data with certain ratio from each label
   :param x_whole: A matrix in which rows correspond to feature vectors
   :param y_whole: A vector containing labels
   :return: training and test data/label
   """
   x_whole = np.concatenate((x_train,x_test), axis=0)
   y_whole = np.concatenate((y_train,y_test), axis=0)

   # Ratio between number of training and test data
   ratio = 0.15

   # All possible labels
   all_labels = np.unique(y_whole)

   # Calculate the number of data available for each label
   data_per_label = {}
   for label in all_labels:
       data_per_label[label] = np.sum(y_whole == label)

   # Initialization
   training_data = np.zeros(shape=(1,x_whole.shape[1]))
   training_label = []
   test_data = np.zeros(shape=(1,x_whole.shape[1]))
   test_label = []

   # Fill in
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

   return training_data, training_label, test_data, test_label

def _read_pepper_cnn(fname):
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


def _optimization(dataset1, dataset2, nb_epochs=3000):
    """
    First trains a model on dataset1, then predicts the labels for vectors in dataset2 using labels of dataset1
    :param dataset1: A dictionary of certain format
    :param dataset2: A dictionary of certain format
    :return: Predicted labels for dataset2 using labels of dataset1
    """

    # x1_mean = dataset1['train'].mean()
    # x1_std = dataset1['train'].std()
    # x_train1 = (dataset1['train'] - x1_mean) / (x1_std)
    x_train1 = dataset1['train']
    y_train1 = dataset1['labels']['train']
    Y_train1 = dataset1['hot_labels']['train']

    # x1_mean = dataset1['test'].mean()
    # x1_std = dataset1['test'].std()
    # x_test1 = (dataset1['test'] - x1_mean) / (x1_std)
    x_test1 = dataset1['test']
    y_test1 = dataset1['labels']['test']
    Y_test1 = dataset1['hot_labels']['test']

    # x2_mean = dataset2['train'].mean()
    # x2_std = dataset2['train'].std()
    # x_train2 = (dataset2['train'] - x2_mean) / (x2_std)
    x_train2 = dataset2['train']

    # x2_mean = dataset2['test'].mean()
    # x2_std = dataset2['test'].std()
    # x_test2 = (dataset2['test'] - x2_mean) / (x2_std)
    x_test2 = dataset2['test']

    batch_size = min(x_train1.shape[0]/7, 16)

    num_labels = len( np.unique( np.concatenate((y_train1,y_test1),axis=0) ) )

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

    hist = model1.fit(x_train1, Y_train1, batch_size=batch_size, nb_epoch=nb_epochs, verbose=1,
                      validation_data=(x_test1, Y_test1), shuffle=True, callbacks=[reduce_lr])

    dataset2_new_labels_train = []

    for i in range(x_train2.shape[0]):
        xTrain = x_train2[i,:].reshape((1,x_train2.shape[1]))
        dataset2_new_labels_train.append(np.argmax(model1.predict(xTrain, batch_size=1)))

    dataset2_new_labels_test = []

    for i in range(x_test2.shape[0]):
        xTest = x_test2[i,:].reshape((1,x_test2.shape[1]))
        dataset2_new_labels_test.append(np.argmax(model1.predict(xTest, batch_size=1)))

    # log = pd.DataFrame(hist.history)
    # with open('Text_Files/' + fname + '_results.txt', 'w') as text_file:
    #     text_file.write(fname + '<<<=====>>>' + str(max(log.val_acc.values)))

    x_model1 = []
    y_model1 = []
    out_model1 = []
    model1 = []

    return dataset2_new_labels_train, dataset2_new_labels_test


def _load_obj(filename):
    with open('/home/ubuntu/big_disk/pickles/' + filename + '.pkl') as f:
        return pickle.load(f)

def _readucr(filename):
    try:
       data = np.loadtxt(filename, delimiter = ',')
       Y = data[:,0]
       X = data[:,1:]
    except:
       data = _load_obj(filename + '.pkl')
       data = np.array([[d[0]]+d[1] for d in data])
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

def _ucr_to_dictionary(x_train, y_train, x_test, y_test):
    #x_train, y_train = _readucr(fname + '_GADF_CNNOUT_' + 'TRAIN')
    #x_test, y_test = _readucr(fname + '_GADF_CNNOUT_' + 'TEST')

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
    print "nb_classes: ", nb_classes
    #y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)
    #y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)

    Y_train = _one_hot_encode(y_train, nb_classes)
    Y_test = _one_hot_encode(y_test, nb_classes)

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

    _THRESHOLD1 = 0.3

    _THRESHOLD2 = 0.3

    # Number of labels in dataset 1
    _NUM_LABELS_D1 = len(np.unique(np.concatenate((dataset1['labels']['train'],dataset1['labels']['test']),axis=0)))

    # Number of labels in dataset 2
    _NUM_LABELS_D2 = len(np.unique(np.concatenate((dataset2['labels']['train'],dataset2['labels']['test']),axis=0)))

    # Call the optimization function to train on the first dataset and predict on the second dataset
    ds2_labels_using_ds1_train, ds2_labels_using_ds1_test = _optimization(dataset1, dataset2, nb_epochs=NUM_EPOCHS)

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

    # Call the optimization function to train on the second dataset and predict on the first dataset
    ds1_labels_using_ds2_train, ds1_labels_using_ds2_test = _optimization(dataset2, dataset1, nb_epochs=NUM_EPOCHS)

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
            print all_labels[i,:]
            print 'label = {0}'.format(label)
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


    # Return the merged dataset as a dictionary
    return new_dataset, Matrix1, Matrix2

# ************************************************* Merge two Datasets *************************************************

# dataset_list = ['Adiac', 'Beef', 'CBF', 'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee', 'Cricket_X', 'Cricket_Y',
#                 'Cricket_Z', 'DiatomSizeReduction', 'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR', '50words', 'FISH',
#                 'Gun_Point', 'Haptics', 'InlineSkate', 'ItalyPowerDemand', 'Lighting2', 'Lighting7', 'MALLAT',
#                 'MedicalImages','MoteStrain', 'NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2', 'OliveOil',
#                 'OSULeaf','SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves', 'SwedishLeaf', 'Symbols',
#                 'synthetic_control', 'Trace', 'TwoLeadECG', 'Two_Patterns', 'uWaveGestureLibrary_X',
#                 'uWaveGestureLibrary_Y','uWaveGestureLibrary_Z', 'wafer', 'WordsSynonyms', 'yoga']

# flist = ['Cricket_X', 'Cricket_Y', 'Cricket_Z']
# flist = ['SonyAIBORobotSurfaceII', 'SonyAIBORobotSurface']
# flist = ['Cricket_X', 'Lighting7', 'SonyAIBORobotSurface']
# flist = ['Lighting7','Lighting2']
#flist = ['FISH', 'InlineSkate', 'Haptics']
# flist = ['Cricket_Z', 'Cricket_X', 'Cricket_Y']
# flist = ['Haptics', 'FISH', 'InlineSkate']
#flist = ['FISH', 'InlineSkate', 'Haptics']

rlist = ['rubicon-fmaq']
# rlist = ['gamut-prod', 'rubicon-fmaq']
rlist = ['rubicon-fmaq', 'gamut-prod']
# x_train, y_train = _readucr('/home/ubuntu/big_disk/GADF_images/' + fname + '_GADF_CNNOUT_' + 'TRAIN')
# x_test, y_test = _readucr('/home/ubuntu/big_disk/GADF_images/' + fname + '_GADF_CNNOUT_' + 'TEST')
x_train, y_train, x_test, y_test = _read_pepper_cnn(rlist[0])
x_train, y_train, x_test, y_test = _split_data_fairly(x_train, y_train, x_test, y_test)
dataset = _ucr_to_dictionary(x_train, y_train, x_test, y_test)

for realm in rlist[1:]:
    x_train, y_train, x_test, y_test = _read_pepper_cnn(realm)
    x_train, y_train, x_test, y_test = _split_data_fairly(x_train, y_train, x_test, y_test)
    dataset1 = _ucr_to_dictionary(x_train, y_train, x_test, y_test)
    dataset, matrix1, matrix2 = _merge_datasets(dataset,dataset1)


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

labels_to_remove = []

if labels_to_remove:

    remaining_rows_train = []
    remaining_rows_test = []

    for row in range(x_train.shape[0]):
        if y_train[row] not in labels_to_remove:
            remaining_rows_train.append(row)

    x_train = x_train[remaining_rows_train,:]
    y_train = y_train[remaining_rows_train]

    for row in range(x_test.shape[0]):
        if y_test[row] not in labels_to_remove:
            remaining_rows_test.append(row)

    x_test = x_test[remaining_rows_test,:]
    y_test = y_test[remaining_rows_test]

    for lbl in labels_to_remove:
        for i in range(len(y_train)):
            if y_train[i] > lbl:
                y_train[i] -= 1

        for i in range(len(y_test)):
            if y_test[i] > lbl:
                y_test[i] -= 1
    num_labels = int(max(max(y_train),max(y_test))+1)

    Y_train = _one_hot_encode(y_train, num_labels)

    Y_test = _one_hot_encode(y_test, num_labels)

    print 'Number of labels after removal: ', num_labels

num_labels = int(max(max(y_train),max(y_test))+1)

batch_size = min(x_train.shape[0]/7, 16)


x_model = Input(x_train.shape[1:])
y_model = Dropout(0.1)(x_model)
y_model = Dense(50, activation='relu')(x_model)
y_model = Dropout(0.2)(y_model)
y_model = Dense(50, activation='relu')(y_model)
out_model = Dense(num_labels, activation='softmax')(y_model)

model = Model(input=x_model, output=out_model)

optimizer = keras.optimizers.Adadelta()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=200, min_lr=0.1)

hist = model.fit(x_train, Y_train, batch_size=batch_size, nb_epoch=NUM_EPOCHS+50, verbose=1,
                 validation_data=(x_test,Y_test), shuffle=True, callbacks=[reduce_lr])

print 'Number of time series after removing unnecessary labels in training data is = {0}'.format(x_train.shape[0])

print 'Number of time series after removing unnecessary labels in test data is = {0}'.format(x_test.shape[0])

print 'Matrix1 = {0}'.format(matrix1)

print 'Matrix2 = {0}'.format(matrix2)

print 'Number of labels is = {0}'.format(num_labels)

print 'Predicted labels vs True labels'
y_pred = []

for i in range(x_test.shape[0]):
    xTest = x_test[i,:].reshape((1,2048))
    print((np.argmax(model.predict(xTest, batch_size=1)),y_test[i]))
    y_pred.append(int(np.argmax(model.predict(xTest, batch_size=1))))

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

# y_test = y_test.tolist()
# y_pred = y_pred.tolist()
if num_labels == 2:
    y_test = y_test.tolist()
    #print 'Precision was {0}.'.format(precision(y_test,y_pred))
    #print 'Recall was {0}.'.format(recall(y_test,y_pred))
    #print 'F1 was {0}.'.format(fbeta_score(y_test,y_pred))

print 'Accuracy per label:'
pprint(accuracy_per_label)

print rlist
