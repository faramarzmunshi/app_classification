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
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from collections import Counter

def _load_obj(filename):
    with open('/home/ubuntu/bigdisk/pickles/' + filename) as f:
        return pickle.load(f)

def read_pepper_cnn(fname):
    print "Reading and formatting data for ", fname
    x_train = np.array(_load_obj(fname + '^cnn_training_output'))[:,0,:]
    x_test = np.array(_load_obj(fname + '^cnn_test_output'))[:,0,:]
    y_train = np.array(_load_obj(fname + '^training_labels'))
    y_test = np.array(_load_obj(fname + '^test_labels'))
    return x_train, y_train, x_test, y_test

y_pred = []

REALMS = ['dataminesjc', 'nbcuni-centralperk2', 'rubicon-fmap', 'gamut-prod', 'rubicon-fbmp', 'nbcuni-superstore', 'jumpshot-jsc', 'rubicon-fbmq', 'rally-health-integration', 'prudential-nj-exp2', 'rubicon-fmaq']

for REALM in REALMS:

	x_train, y_train, x_test, y_test = read_pepper_cnn(REALM)

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

	num_labels = int(max(max(y_train),max(y_test))+1)

	batch_size = min(x_train.shape[0]/7, 16)

	model = load_model('../FINAL')

	y_whole = np.concatenate((y_train,y_test), axis=0)
	x_whole = np.concatenate((x_train,x_test), axis=0)

	for i in range(x_whole.shape[0]):
	    xTest = x_whole[i,:].reshape((1,2048))
	    y_pred.append(int(np.argmax(model.predict(xTest, batch_size=1))))

	#pprint(dict(Counter(tuple(reversed(y)) for y in y_pred)))
pprint(dict(Counter(y_pred)))
	#y_pred = []

	#for i in range(x_train.shape[0]):
	#    xTrain = x_train[i,:].reshape((1,2048))
	#    y_pred.append((int(np.argmax(model.predict(xTrain, batch_size=1))),y_train[i]))

	#pprint(dict(Counter(tuple(reversed(y)) for y in y_pred)))
	# pprint(dict(Counter(y_pred)))

	#y_pred = y_pred.tolist()
	#accuracy_per_label = {}
	#y_pred = np.array(y_pred)
	#y_test = np.array(y_whole)
	#for i in range(num_labels):
	#    pred_label_loc = np.where(y_pred==i)[0]
	#    actual_label_loc = np.where(y_test==i)[0]
	#    accuracy_per_label[i] = 0
	#    for j in pred_label_loc:
	#        if j in actual_label_loc:
	#            accuracy_per_label[i] += 1
	#    if actual_label_loc.tolist():
	#        accuracy_per_label[i] = (accuracy_per_label[i]*100.0/len(actual_label_loc), len(actual_label_loc))

	#print 'Accuracy per label:'
	#pprint(accuracy_per_label)
