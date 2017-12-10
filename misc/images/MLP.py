#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 21:46:23 2016

@author: stephen
"""
 
from __future__ import print_function
import numpy as np 
# np.random.seed(813306)
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.utils import np_utils
import keras 
from keras.callbacks import ReduceLROnPlateau
import pandas as pd
import pickle
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def _load_obj(filename):
    with open(filename) as f:
	return pickle.load(f)
      
def readucr(filename):
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
  
nb_epochs = 3000
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

for each in flist:
    fname = each
    print("Currently looking at" + fname)
    x_train, y_train = readucr(prefix + fname + '_' + _MODE + '_CNNOUT_TRAIN')
    x_test, y_test = readucr(prefix + fname + '_' + _MODE + '_CNNOUT_TEST')
    nb_classes =len(np.unique(y_test))
    y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
    y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)
    batch_size = min(x_train.shape[0]/10, 16)
    
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
     
    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train = (x_train - x_train_mean)/(x_train_std)
     
    # x_test_min = np.min(x_test, axis = 1, keepdims=1)
    # x_test_max = np.max(x_test, axis = 1, keepdims=1)
    x_test = (x_test - x_train_mean)/(x_train_std)
     
    # x_train = x_train.reshape(x_train.shape + (1,))
    # x_test = x_test.reshape(x_test.shape + (1,))
    
    x = Input(x_train.shape[1:])
    y = Dropout(0.1)(x)
    y = Dense(100, activation='relu')(x)
    #y = Dropout(0.2)(y)
    #y = Dense(50, activation='relu')(y)
    out = Dense(nb_classes, activation='softmax')(y)
     
    model = Model(input=x, output=out)
     
    optimizer = keras.optimizers.Adadelta()    
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
     
    reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor=0.5,
                      patience=200, min_lr=0.1)
    
    hist = model.fit(x_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
              verbose=1, validation_data=(x_test, Y_test), shuffle=False, callbacks=[reduce_lr])
    
    #Print the testing results which has the lowest training loss.
    log = pd.DataFrame(hist.history)
    print("saving results for 100 nodes" + _MODE + fname)
    log.to_json('accuracies/accuracy_100_' + _MODE + fname + '.json')

 

