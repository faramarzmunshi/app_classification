#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 20:11:19 2016

@author: stephen
"""
 
from __future__ import print_function
 
from keras.models import Model
from keras.utils import np_utils
import numpy as np
import pandas as pd

import keras 
from keras.callbacks import ReduceLROnPlateau
      
def readucr(filename):
    data = np.loadtxt('UCR/' + filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

def normalization(training_data):
    maximum = np.max(training_data)
    minimum = np.min(training_data)
    out = training_data * 2
    out -= maximum + minimum
    out /= (maximum - minimum)
    return out

def phi_matrix(training_data):
    return np.arccos(training_data)

def GADF_image(phi_vector):
    temp_sin = np.sin(phi_vector).reshape((len(phi_vector),1))
    temp_cos = np.cos(phi_vector).reshape((len(phi_vector),1))
    out = np.dot(temp_sin,temp_cos.T) - np.dot(temp_cos,temp_sin.T)
    return out


nb_epochs = 2000


#flist = ['Adiac', 'Beef', 'CBF', 'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 
#'DiatomSizeReduction', 'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR', '50words', 'FISH', 'Gun_Point', 'Haptics', 
#'InlineSkate', 'ItalyPowerDemand', 'Lighting2', 'Lighting7', 'MALLAT', 'MedicalImages', 'MoteStrain', 'NonInvasiveFatalECG_Thorax1', 
#'NonInvasiveFatalECG_Thorax2', 'OliveOil', 'OSULeaf', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves', 'SwedishLeaf', 'Symbols', 
#'synthetic_control', 'Trace', 'TwoLeadECG', 'Two_Patterns', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'wafer', 'WordsSynonyms', 'yoga']

flist  = ['Beef']

# for each in flist:
#     fname = each
#     x_train, y_train = readucr(fname + '/' + fname + '_TRAIN')
#     x_test, y_test = readucr(fname + '/' + fname + '_TEST')
#
#     nb_classes = len(np.unique(y_test))
#
#     y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)
#     y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)
#
#     Y_train = np_utils.to_categorical(y_train, nb_classes)
#     Y_test = np_utils.to_categorical(y_test, nb_classes)
#
#     x_train = normalization(x_train)
#     x_test = normalization(x_test)
#
#     x_train = phi_matrix(x_train)
#     x_test = phi_matrix(x_test)
#
#     for i in range(x_train.shape[0]):
#         img = GADF_image(x_train[i,:])
#         np.savetxt('Images/train' + fname + str(i),img)
#
#     for i in range(x_test.shape[0]):
#         img = GADF_image(x_test[i,:])
#         np.savetxt('Images/test' + fname + str(i),img)
#
#
#     np.savetxt('Images/training_label' + fname,y_train)
#     np.savetxt('Images/training_hot_label' + fname, Y_train)
#     np.savetxt('Images/test_labels' + fname, y_test)
#     np.savetxt('Images/test_hot_labels' + fname, Y_test)
#
#
#
# assert 2 == 1


for each in flist:
    fname = each
    x_train, y_train = readucr(fname+'/'+fname+'_TRAIN')
    x_test, y_test = readucr(fname+'/'+fname+'_TEST')
    nb_classes = len(np.unique(y_test))
    batch_size = min(x_train.shape[0]/10, 16)
    
    y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
    y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)
    
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    training_images = np.zeros(shape=(x_train.shape[0],x_train.shape[1],x_train.shape[1],1))
    test_images = np.zeros(shape=(x_test.shape[0],x_test.shape[1],x_test.shape[1],1))

    x_train = normalization(x_train)
    x_test = normalization(x_test)

    x_train = phi_matrix(x_train)
    x_test = phi_matrix(x_test)

    for i in range(x_train.shape[0]):
        training_images[i,:,:,0] = GADF_image(x_train[i,:])

    print('Train images are created!')

    for i in range(x_test.shape[0]):
        test_images[i,:,:,0] = GADF_image(x_test[i,:])


    print('Test images are created!')

    # x_train_mean = x_train.mean()
    # x_train_std = x_train.std()
    # x_train = (x_train - x_train_mean)/(x_train_std)
    #
    # x_test = (x_test - x_train_mean)/(x_train_std)
    # x_train = x_train.reshape(x_train.shape + (1,1,))
    # x_test = x_test.reshape(x_test.shape + (1,1,))

    x = keras.layers.Input(training_images.shape[1:])
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
     
    reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor=0.5, patience=50, min_lr=0.0001)
    hist = model.fit(training_images, Y_train, batch_size=batch_size, nb_epoch=nb_epochs, verbose=1,
                     validation_data=(test_images, Y_test), callbacks = [reduce_lr])
    #Print the testing results which has the lowest training loss.
    log = pd.DataFrame(hist.history)
    #print log.loc[log[‘loss'].idxmin]['loss’], log.loc[log[‘loss'].idxmin][‘val_acc’]
