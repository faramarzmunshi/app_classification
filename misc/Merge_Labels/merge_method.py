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


NUM_EPOCHS = 1500
# **************************************************** Functions *******************************************************
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

    x1_mean = dataset1['data'].mean()
    x1_std = dataset1['data'].std()
    x1 = (dataset1['data'] - x1_mean) / (x1_std)
    y1 = dataset1['labels']
    Y1 = dataset1['hot_labels']

    x2_mean = dataset2['data'].mean()
    x2_std = dataset2['data'].std()
    x2 = (dataset2['data'] - x2_mean) / (x2_std)

    x_model1 = Input(x1.shape[1:])
    y_model1 = Dropout(0.1)(x_model1)
    y_model1 = Dense(50, activation='relu')(x_model1)
    y_model1 = Dropout(0.2)(y_model1)
    y_model1 = Dense(50, activation='relu')(y_model1)
    out_model1 = Dense(len(np.unique(y1)), activation='softmax')(y_model1)

    model1 = Model(input=x_model1, output=out_model1)

    optimizer = keras.optimizers.Adadelta()
    model1.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=200, min_lr=0.1)

    hist = model1.fit(x1, Y1, batch_size=x1.shape[0], nb_epoch=nb_epochs, verbose=1, shuffle=True, callbacks=[reduce_lr])

    dataset2_new_labels = []

    for i in range(x2.shape[0]):
        xTrain = x2[i,:].reshape((1,x2.shape[1]))
        dataset2_new_labels.append(np.argmax(model1.predict(xTrain, batch_size=1)))

    # Print the testing results which has the l in range(x_train.shape[0]):
    #    for i in range(len(x_test1)):
    #        xTest = x_test1[i,:].reshape((1,2048))
    #        print((np.argmax(model.predict(xTest, batch_size=1)), y_test1[i]))
    # log = pd.DataFrame(hist.history)
    # print("saving results for 100 nodes" + _MODE + fname)
    # log.to_json('accuracies/accuracy_100_' + _MODE + fname + '.json')

    # with open('Text_Files/' + fname + '_results.txt', 'w') as text_file:
    #     text_file.write(fname + '<<<=====>>>' + str(max(log.val_acc.values)))

    # assert 2==1

    x_model1 = []
    y_model1 = []
    out_model1 = []
    model1 = []

    return dataset2_new_labels


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


def _ucr_to_dictionary(fname):
    x_train, y_train = _readucr('/home/ubuntu/big_disk/GADF_images/' + fname + '_GADF_CNNOUT_' + 'TRAIN')
    x_test, y_test = _readucr('/home/ubuntu/big_disk/GADF_images/' + fname + '_GADF_CNNOUT_' + 'TEST')
    #x_train, y_train = _readucr(fname + '_GADF_CNNOUT_' + 'TRAIN')
    #x_test, y_test = _readucr(fname + '_GADF_CNNOUT_' + 'TEST')

    nb_classes = len(np.unique(y_test))

    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    dictionary = {'actual_lengths': {}}

    dictionary['data'] = np.concatenate((x_train,x_test), axis=0)
    dictionary['labels'] = np.concatenate((y_train,y_test), axis=0)
    dictionary['hot_labels'] = np.concatenate((Y_train,Y_test), axis=0)
    dictionary['actual_lengths']['data'] = np.concatenate((np.zeros(shape=(x_train.shape[0])) + x_train.shape[1],
                                                           np.zeros(shape=(x_test.shape[0])) + x_test.shape[1]),
                                                          axis=0)

    return dictionary


def _merge_datasets(dataset1, dataset2):
    """
    Merge two datasets to unify their labels
    :param dataset1: A dataset of certain format                                   @vtype: A dictionary
    :param dataset2: A dataset of certain format                                   @vtype: A dictionary
    :return: A merged dataset with unified labels of the same format as inputs     @vtype: A dictionary
    """

    # Number of labels in dataset 1
    _NUM_LABELS_D1 = len(np.unique(dataset1['labels']))

    # Number of labels in dataset 2
    _NUM_LABELS_D2 = len(np.unique(dataset2['labels']))

    # Call the optimization function to train on the first dataset and predict on the second dataset
    ds2_labels_using_ds1 = _optimization(dataset1, dataset2, nb_epochs=NUM_EPOCHS)

    # Initialize the label counting matrix
    label_counter = np.zeros(shape=(_NUM_LABELS_D2, _NUM_LABELS_D1))

    # Fill the label counting matrix accordingly
    for i in range(len(ds2_labels_using_ds1)):
        label_counter[int(dataset2['labels'][i]), int(ds2_labels_using_ds1[i])] += 1

    Matrix1 = np.matrix.copy(label_counter)

    # Initialize the new set of labels for dataset 2
    ds2_new_labels = np.zeros(shape=(len(ds2_labels_using_ds1), 2))

    # Determine the new labels for dataset 2
    for i in range(len(ds2_labels_using_ds1)):
        if dataset2['labels'][i] == np.argmax(label_counter[:, int(ds2_labels_using_ds1[i])]):
            ds2_new_labels[i, :] = np.array([ds2_labels_using_ds1[i], dataset2['labels'][i]])
        else:
            ds2_new_labels[i, :] = np.array([ds2_labels_using_ds1[i], -1])

    # Call the optimization function to train on the second dataset and predict on the first dataset
    ds1_labels_using_ds2 = _optimization(dataset2, dataset1, nb_epochs=NUM_EPOCHS)

    # Initialize the label counting matrix
    label_counter = np.zeros(shape=(_NUM_LABELS_D1, _NUM_LABELS_D2))

    # Fill the label counting matrix accordingly
    for i in range(len(ds1_labels_using_ds2)):
        label_counter[int(dataset1['labels'][i]), int(ds1_labels_using_ds2[i])] += 1

    Matrix2 = np.matrix.copy(label_counter.T)

    # Initialize the new set of labels for dataset 1
    ds1_new_labels = np.zeros(shape=(len(ds1_labels_using_ds2), 2))

    # Determine the new labels for dataset 1
    for i in range(len(ds1_labels_using_ds2)):
        if ds1_labels_using_ds2[i] == np.argmax(label_counter[int(dataset1['labels'][i]), :]):
            ds1_new_labels[i, :] = np.array([dataset1['labels'][i], ds1_labels_using_ds2[i]])
        else:
            ds1_new_labels[i, :] = np.array([dataset1['labels'][i], -1])

    # Concatenate all labels from both datasets
    all_labels = np.concatenate((ds1_new_labels, ds2_new_labels), axis=0)

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

    # Initialize the concatenated dataset
    new_dataset = {'labels': vector_label, 'hot_labels': hot_labels, 'actual_lengths': {}}

    # Fill the corresponding keys for the concatenated dataset
    for key in dataset1.keys():
        if (key != 'labels') and (key != 'hot_labels') and (key != 'actual_lengths'):
            new_dataset[key] = np.concatenate((dataset1[key], dataset2[key]), axis=0)
        if key == 'actual_lengths':
            for key2 in dataset1[key]:
                new_dataset[key][key2] = np.concatenate((dataset1[key][key2], dataset2[key][key2]), axis=0)

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

flist = ['Cricket_X', 'Cricket_Y', 'Cricket_Z']
flist = ['SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII']
#flist = ['SonyAIBORobotSurface', 'Cricket_X', 'Lighting7']
# flist = ['Lighting2','Lighting7']
# flist = ['FISH', 'Haptics', 'InlineSkate']

dataset = _ucr_to_dictionary(flist[0])

for num_fname in range(1,len(flist)):
    dataset1 = _ucr_to_dictionary(flist[num_fname])
    dataset, matrix1, matrix2 = _merge_datasets(dataset,dataset1)

x = dataset['data']
y = dataset['labels']
Y = dataset['hot_labels']

label_index = {}

for i in range(len(np.unique(y))):
    label_index[i] = np.sum(y == i)

print 'Number of time series per label is = {0}'.format(label_index)

print 'Number of time series is = {0}'.format(x.shape[0])

labels_to_remove = []

remaining_rows = []

for row in range(x.shape[0]):
    if y[row] not in labels_to_remove:
        remaining_rows.append(row)

x = x[remaining_rows,:]
y = y[remaining_rows]
Y = Y[remaining_rows,:]

print 'Number of time series after removing unnecessary labels is = {0}'.format(x.shape[0])

for lbl in labels_to_remove:
    for i in range(len(y)):
        if y[i] > lbl:
            y[i] -= 1

Y = _one_hot_encode(y, len(np.unique(y)))

x_mean = dataset['data'].mean()
x_std = dataset['data'].std()
x = (dataset['data'] - x_mean) / (x_std)

ind = range(len(remaining_rows))

random.shuffle(ind)

NUM_TEST = int(x.shape[0]/7)
x_test = x[ind[0:NUM_TEST],:]
y_test = y[ind[0:NUM_TEST]]
Y_test = Y[ind[0:NUM_TEST],:]

x_train = x[ind[NUM_TEST:],:]
y_train = y[ind[NUM_TEST:]]
Y_train = Y[ind[NUM_TEST:],:]

print 'number of labels: ',len(np.unique(y))

x_model = Input(x.shape[1:])
y_model = Dropout(0.1)(x_model)
y_model = Dense(50, activation='relu')(x_model)
y_model = Dropout(0.2)(y_model)
y_model = Dense(50, activation='relu')(y_model)
out_model = Dense(len(np.unique(y)), activation='softmax')(y_model)

model = Model(input=x_model, output=out_model)

optimizer = keras.optimizers.Adadelta()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=200, min_lr=0.1)

hist = model.fit(x_train, Y_train, batch_size=len(remaining_rows), nb_epoch=NUM_EPOCHS+1000, verbose=1, validation_data=(x_test,Y_test),
                 shuffle=True, callbacks=[reduce_lr])

print 'Matrix1 = {0}'.format(matrix1)

print 'Matrix2 = {0}'.format(matrix2)

print 'Number of time series per label is = {0}'.format(label_index)

num_labels = len(np.unique(y))
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

if num_labels == 2:
    y_test = y_test.tolist()
    print 'Precision was {0}.'.format(precision(y_test,y_pred))
    print 'Recall was {0}.'.format(recall(y_test,y_pred))
    print 'F1 was {0}.'.format(fbeta_score(y_test,y_pred))

print 'Accuracy per label:'
pprint(accuracy_per_label)

    
