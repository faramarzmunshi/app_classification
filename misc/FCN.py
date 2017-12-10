import numpy as np
import random

np.random.seed(813306)

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

NUM_EPOCHS = 5000

def _load_obj(filename):
    with open('/home/ubuntu/bigdisk/pickles/' + filename) as f:
        return pickle.load(f)

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



def _split_data_fairly(x_train, y_train, x_test, y_test):
    """
    Split whole data into training and test data with certain ratio from each label
    :param x_whole: A matrix in which rows correspond to feature vectors
    :param y_whole: A vector containing labels
    :return: training and test data/label
    """
    print "Splitting data fairly..."
    x_whole = np.concatenate((x_train,x_test), axis=0)
    y_whole = np.concatenate((y_train,y_test), axis=0)

    MIN_NUM_PER_LABEL = 60
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
    # Fill in
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

    print "Removed labels less than ", MIN_NUM_PER_LABEL

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
    print "Reading and formatting data for ", fname
    x_train = np.array(_load_obj(fname + '^cnn_training_output'))[:,0,:]
    x_test = np.array(_load_obj(fname + '^cnn_test_output'))[:,0,:]
    y_train = np.array(_load_obj(fname + '^training_labels'))
    y_test = np.array(_load_obj(fname + '^test_labels'))
    return x_train, y_train, x_test, y_test

rlist  = ['gamut-prod']
for realm in rlist:
    x_train, y_train, x_test, y_test = _read_pepper_cnn(realm)
    x_train, y_train, x_test, y_test = _split_data_fairly(x_train, y_train, x_test, y_test)
    batch_size = min(x_train.shape[0]/10, 100)
    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
    Y_train = _one_hot_encode(y_train, nb_classes)
    Y_test = _one_hot_encode(y_test, nb_classes)

    x_train = x_train.reshape(x_train.shape + (1,1,))
    x_test = x_test.reshape(x_test.shape + (1,1,))    
    x = keras.layers.Input(x_train.shape[1:])
#    drop_out = Dropout(0.2)(x)
    conv1 = keras.layers.Conv2D(128, 8, 1, border_mode='same')(x)
    conv1 = keras.layers.normalization.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation('relu')(conv1)
    
#    drop_out = Dropout(0.2)(conv1)
    conv2 = keras.layers.Conv2D(256, 5, 1, border_mode='same')(conv1)
    conv2 = keras.layers.normalization.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)
    
#    drop_out = Dropout(0.2)(conv2)
    conv3 = keras.layers.Conv2D(128, 3, 1, border_mode='same')(conv2)
    conv3 = keras.layers.normalization.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)
    
    full = keras.layers.pooling.GlobalAveragePooling2D()(conv3)    
    out = keras.layers.Dense(nb_classes, activation='softmax')(full)
    
    
    model = Model(input=x, output=out)
     
    optimizer = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
     
    reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor=0.5,
                      patience=50, min_lr=0.0001) 
    hist = model.fit(x_train, Y_train, batch_size=batch_size, nb_epoch=NUM_EPOCHS,
              verbose=1, validation_data=(x_test, Y_test), callbacks = [reduce_lr])
    #Print the testing results which has the lowest training loss.
    log = pd.DataFrame(hist.history)
    print log.loc[log['loss'].idxmin]['loss'], log.loc[log['loss'].idxmin]['val_acc']
