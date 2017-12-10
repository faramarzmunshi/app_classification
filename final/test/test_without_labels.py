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
import json

def _load_obj(filename):
    with open('/bigdisk/testing_pickles/' + filename) as f:
        return pickle.load(f)

def read_pepper_cnn(fname):
    print "Reading and formatting data for ", fname
    x_train = np.array(_load_obj(fname + '^cnn_training_output'))[:,0,:]
    x_test = np.array(_load_obj(fname + '^cnn_test_output'))[:,0,:]
    y_train = np.array(_load_obj(fname + '^training_labels'))
    y_test = np.array(_load_obj(fname + '^test_labels'))
    return x_train, y_train, x_test, y_test

REALM = 'rubicon-fmap'

x_whole = _load_obj(REALM + '^cnn_vectors')
app_ids = _load_obj(REALM + '^app_ids')

batch_size = min(len(x_whole)/7, 16)

model = load_model('../FINAL_MERGING_NEW_NAMES_0TH')

y_pred = {}

for row_num, vector in enumerate(x_whole):
    vector = np.array(vector).reshape(1,2048)
    pred = np.argmax(model.predict(vector, batch_size=1))
    if pred in y_pred.keys():
        y_pred[pred].append(app_ids[row_num])
    else:
        y_pred[pred] = [app_ids[row_num]]


with open('results.txt', 'w') as f:
    json.dump(y_pred, f, indent=4)

