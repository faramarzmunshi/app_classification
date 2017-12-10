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
import re

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


REALM = 'gamut-prod'
app_ids = _load_obj(REALM + '^app_ids')
names = [x[1] for x in app_ids]
changed = []
for name in names:
    changed_name = re.sub('[0-9][0-9][0-9]+','',name)
    changed_name = re.sub('__.*','',changed_name)
    changed_name = re.sub('_.*','',changed_name)
#    changed_name = re.sub('\W+','', changed_name).split('_')[0]
    #changed_name = changed_name.split('_')[0]
#    print (name, changed_name)
    changed.append(changed_name)

print len(np.unique(changed))
l = []
for key, value in dict(Counter(changed)).iteritems():
    l.append((key,value))

pprint(sorted(l, key=lambda x: x[1]))
