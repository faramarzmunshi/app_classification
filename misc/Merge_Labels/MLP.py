#usr/bin/env python2
# -*- coding: utf-    8 -*-
"""
Created on Fri Oct 28 21:46:23 2016

@author: stephen
"""
 
import numpy as np 
# np.random.seed(813306)
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.utils import np_utils
import keras 
from keras.callbacks import ReduceLROnPlateau
import pandas as pd
import pickle
from collections import Counter
import os
from pprint import pprint
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def _one_hot_encode(label_vector, total_num_labels):
    """
    One hot encode a label vector.
    :param label_vector: a vector of labels for each time series
    :param total_num_labels: total number of labels
    :return: one hot encoded version of labels of all time series
    """
    out = np.zeros(shape=(len(label_vector), total_num_labels))
    for i in range(len(label_vector)):
        out[i, label_vector[i]] = 1
    return out

def _load_obj(filename):
    with open('/home/ubuntu/big_disk/strip_images/' + filename) as f:
	return pickle.load(f)
      
def readucr(filename):
    try:
	data = np.loadtxt(filename, delimiter = ',')
    except:
	data = _load_obj(filename)
	data = np.array([[d[0]]+d[1] for d in data])
	 
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

def read_pepper(fname):
    x_train = np.array(_load_obj(fname + '^cnn_training_output'))[:,0,:]
    x_test = np.array(_load_obj(fname + '^cnn_test_output'))[:,0,:]
    y_train = np.array(_load_obj(fname + '^training_labels'))
    y_test = np.array(_load_obj(fname + '^test_labels'))
    return x_train, x_test, y_train, y_test

nb_epochs = 1000
_MODE = 'strip' # 'strip'
if _MODE == 'GADF':
    prefix = 'GADF_images/'
elif _MODE == 'strip':
    prefix = 'strip_images/'
    
flist = ['Adiac', 'Beef', 'CBF', 'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 
'DiatomSizeReduction', 'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR', '50words', 'FISH', 'Gun_Point', 'Haptics', 
'InlineSkate', 'ItalyPowerDemand', 'Lighting2', 'Lighting7', 'MALLAT', 'MedicalImages', 'MoteStrain', 'NonInvasiveFatalECG_Thorax1', 
'NonInvasiveFatalECG_Thorax2', 'OliveOil', 'OSULeaf', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves', 'SwedishLeaf', 'Symbols', 
'synthetic_control', 'Trace', 'TwoLeadECG', 'Two_Patterns', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'wafer', 'WordsSynonyms', 'yoga']

#flist = ['Cricket_X', 'Cricket_Y', 'Cricket_Z']
#flist = ['SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII']
#flist = ['SonyAIBORobotSurface', 'Cricket_X', 'Lighting7']
#flist = ['Lighting7', 'Lighting2']
flist = ['FISH', 'Haptics', 'InlineSkate']
x_train = np.zeros((0,2048))
x_test = np.zeros((0,2048))
y_train = []
y_test = []
for fname in flist:
    print("Currently looking at " + fname)
    data_train, label_train = readucr(fname + '_strip_CNNOUT_TRAIN')
    data_test, label_test = readucr(fname + '_strip_CNNOUT_TEST')
    x_train = np.concatenate((data_train, x_train), axis=0)
    x_test = np.concatenate((data_test, x_test), axis=0)
    try:
        maxlabel = max(y_train)
    except:
        maxlabel = 0
    label_train += maxlabel + 1
    label_test += maxlabel + 1
    y_train += label_train.tolist()
    y_test += label_test.tolist()

# TO BE TAKEN OUT
#temp = y_train
#y_train = y_test
#y_test = temp
#temp = x_train
#x_train = x_test
#x_test = temp

y_train = np.array(y_train)
y_test = np.array(y_test)    
y_train1 = y_train
x_train1 = x_train
x_test1 = x_test
y_test1 = y_test
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
nb_classes =len(np.unique(y_test))
print("Number of labels in y_test: {0}".format(nb_classes))
nb_classes =len(np.unique(y_train))
print("Number of labels in y_train: {0}".format(nb_classes))
print("Number per label in y_test: ")
pprint(dict(Counter(y_test)))
print("Number per label in y_train: ")
pprint(dict(Counter(y_train)))
y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)    
batch_size = min(x_train.shape[0]/10, 16)
hot_train = _one_hot_encode(map(int, y_train), nb_classes)
hot_test = _one_hot_encode(map(int, y_test), nb_classes)
    #for index, labels in enumerate(hot_train):
    #    if y_train1[index] != 0:
    #        print(labels, y_train1[index])
    #x_train_mean = x_train.mean()
    #x_train_std = x_train.std()
    #x_train = (x_train - x_train_mean)/(x_train_std)
     
    # x_test_min = np.min(x_test, axis = 1, keepdims=1)
    # x_test_max = np.max(x_test, axis = 1, keepdims=1)
    #x_test = (x_test - x_train_mean)/(x_train_std)
     
    # x_train = x_train.reshape(x_train.shape + (1,))
    # x_test = x_test.reshape(x_test.shape + (1,))
    
x = Input(x_train.shape[1:])
y = Dropout(0.1)(x)
y = Dense(50, activation='relu')(x)
y = Dropout(0.2)(y)
y = Dense(50, activation='relu')(y)
out = Dense(nb_classes, activation='softmax')(y)
     
model = Model(input=x, output=out)
     
optimizer = keras.optimizers.Adadelta() 
model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
     
reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor=0.5,
                      patience=200, min_lr=0.1)
hist = model.fit(x_train, hot_train, batch_size=batch_size, nb_epoch=nb_epochs,
              verbose=1, validation_data=(x_test, hot_test),
              shuffle=True, callbacks=[reduce_lr])

y_pred =[]
for i in range(x_test.shape[0]):
    xTest = x_test[i,:].reshape((1,2048))
    print((np.argmax(model.predict(xTest, batch_size=1)),y_test[i]))
    y_pred.append(int(np.argmax(model.predict(xTest, batch_size=1))))

accuracy_per_label = {}
y_pred = np.array(y_pred)
y_test = np.array(y_test)
for i in range(nb_classes):
    pred_label_loc = np.where(y_pred==i)[0]
    actual_label_loc = np.where(y_test==i)[0]
    accuracy_per_label[i] = 0
    for j in pred_label_loc:
        if j in actual_label_loc:
            accuracy_per_label[i] += 1
    if actual_label_loc.tolist():
        accuracy_per_label[i] = (accuracy_per_label[i]*100.0/len(actual_label_loc), len(actual_label_loc))


    #for i in range(x_train.shape[0]):
        #xTrain = x_train[i,:].reshape((1,2048))
        #print((np.argmax(model.predict(xTrain, batch_size=1)),y_train[i]))

    #Print the testing results which has the l in range(x_train.shape[0]):
#    for i in range(len(x_test1)):
#        xTest = x_test1[i,:].reshape((1,2048))
#        print((np.argmax(model.predict(xTest, batch_size=1)), y_test1[i]))
    # log = pd.DataFrame(hist.history)
    #print("saving results for 100 nodes" + _MODE + fname)
    #log.to_json('accuracies/accuracy_100_' + _MODE + fname + '.json')

pprint(accuracy_per_label) 

