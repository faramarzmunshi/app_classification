
# FOR LOADING DATA AND SAVING IT IN PROPERLY FORMATTED NUMPY ARRAYS

# *********************************************** Imports **************************************************************
import json
# for parsing raw data
import pickle
# for saving data to files
import numpy as np
# for splitting train and test data
import os
# for traversing file paths
# Import a helper-function for caching the transfer-values of the Inception model
from inception import transfer_values_cache
from pprint import pprint
from datetime import datetime
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
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from collections import Counter
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

_METRICS = ['vmsram', 'tasks', 't_rscthnetno', 't_rscthhfsrb', 'c_ucpupct']

# *********************************************** Functions ************************************************************
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

def _split_data_fairly(x_whole, y_whole):
    """
    Split whole data into training and test data with certain ratio from each label
    :param x_whole: A matrix in which rows correspond to feature vectors
    :param y_whole: A vector containing labels
    :return: training and test data/label
    """
    print "Splitting data fairly..."

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
    indexes = range(test_data.shape[0])
    random.shuffle(indexes)
    test_data = test_data[indexes,:]
    test_label = test_label[indexes]

    print "Split data."

    return training_data, training_label, test_data, test_label

def _create_image(list_of_dicts, largest_dim):
    """
    Create a collection of images for each time series across all metrics
    :param list_of_dicts: A list of dictionaries where each dictionary contains phi information about one time series
    :return: @rtype = np.array - Collection of images for those time series
    """
    timer = datetime.now()
    # All possible metrics
    _METRICS = ['vmsram', 'tasks', 't_rscthnetno', 't_rscthhfsrb', 'c_ucpupct']

    # Initialize the collection of all images concatenated across all metrics
    images = np.zeros(shape=(len(list_of_dicts), len(_METRICS) * largest_dim, largest_dim, 1))

    # For each dictionary in the list of dictionaries (i.e., for each time series)
    for num_dict, dict in enumerate(list_of_dicts):
        # For each metric
        for metric_num, metric in enumerate(_METRICS):
            temp_sin = np.sin(list_of_dicts[num_dict][metric][0:list_of_dicts[num_dict]['actual_lengths'][metric]])

            temp_sin = temp_sin.reshape((len(temp_sin), 1))

            temp_cos = np.cos(list_of_dicts[num_dict][metric][0:list_of_dicts[num_dict]['actual_lengths'][metric]])

            temp_cos = temp_cos.reshape((len(temp_cos), 1))

            # Calculate the image for a specific metric and time series
            images[num_dict,
            (metric_num * largest_dim):(metric_num * largest_dim + list_of_dicts[num_dict]['actual_lengths'][metric]),
            0:list_of_dicts[num_dict]['actual_lengths'][metric],
            0] = _interpolation(np.dot(temp_sin,temp_cos.T) - np.dot(temp_cos,temp_sin.T))

    # Scale the set of images to [0,255] interval required by CNN
    images += 1
    images *= 127.5

    return images

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

def _compose_dimension_matrices(data):
    """
    Composes dimension matrices that are needed for smartly batching
    :param training_data: set of training data
    :param test_data: set of test data
    :return: @rtype = np.array - arrays of max length for each time series in both test and training data
    """
    # Number of the training data
    _NUM_DATA = data[_METRICS[0]].shape[0]

    # Number of the test data

    # Dictionary that contains dimensions for each metric for each time series
    dim_matrix = {}

    # The largest length of time series for all metrics in a certain row in the training and test data
    max_dim = []

    # Filling in the dimension matrix with the max length across metrics for a single application
    for row in range(_NUM_DATA):
        for metric in _METRICS:
            dim = data['actual_lengths'][metric][row]
            if dim != 0:
                dim_matrix[metric] = data['actual_lengths'][metric][row]
            else:
                dim_matrix[metric] = 1
        max_dim.append(max(dim_matrix.values()))

    return max_dim

def _create_phi_data(data):
    """
    Creates the phi version of the training and test datasets.
    :param training_data: set of training data for dataset
    Creates the phi version of the training and test datasets.
    :param training_data: set of training data for dataset
    :param test_data: set of test data for dataset
    :return: @rtype = dict - the phi version of the training and test datasets
    """
    phi_data = {}

    # Iterate and compute arccos of each time series in training and test data
    for key in data.keys():
        if key in _METRICS:
            phi_data[key] = np.arccos(data[key])
        else:
            phi_data[key] = data[key]

    return phi_data

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

def _save_obj(obj, name):
    """
    Saves an object in a pickle file.
    :param obj: object to be saved
    :param name: name of file/file path
    """
    directory = '/bigdisk/testing_pickles/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(directory + name, 'w') as outfile:
    	pickle.dump(obj, outfile)

def _compile_filepath_list(input):
        import glob, os
        filepaths = []
        for root, dirs, files in os.walk('./data/'):
            for file in files:
                if file.endswith(input):
                    filepath = os.path.join(root,file)
                    filepaths += [filepath]
        return filepaths

def _image_generator(phi_data, max_dim, labels, nb_classes):
    _NUM_DATA = phi_data[_METRICS[0]].shape[0]
    labels = _one_hot_encode(labels, nb_classes)
#    images, l = [], []
    for row in range(_NUM_DATA):
        temp_dict = {'actual_lengths': {}}
        for metric in _METRICS:
            temp_dict[metric] = phi_data[metric][row]
            temp_dict['actual_lengths'][metric] = phi_data['actual_lengths'][metric][row]
        input_dict = [temp_dict]
        largest_dim = max_dim[row]
        image = _create_image(input_dict, largest_dim)
#        images.append(image)
#        l.append(labels[row].reshape((1,len(labels[row]))))
#    return images, l
        yield image, labels[row].reshape((1,len(labels[row])))

# ************************************* Constants and Variable Loading *************************************************

def main():
    # Name of file to load data from
    filenames = ['./data/rubicon-fmaq_7days_app_saeed_data.json']
    for filename in filenames:
        print "Opened and reading {0}...".format(filename)
        _FILE_NAME = filename

        REALM = _FILE_NAME.split('_')[0].split('/')[-1]

        # List of all metrics collected for each job
        _METRICS = ['vmsram', 'tasks', 't_rscthnetno', 't_rscthhfsrb', 'c_ucpupct']

        _NUM_CHAR = 10

        # Open data file and load data
        with open(_FILE_NAME) as infile:
            data_set = json.load(infile)

        # ************************************* Constitute the label vector ************************************************

        # Initialize the variables
        labels = []
        tags = []
        label_count = {}

        # Iterate through the data set and identify unique labels by the first _NUM_CHAR characters in the name of the jobs
        for data in data_set:
            name = data_set[data]['job_name'].upper()
            label = name[:_NUM_CHAR]
            if label not in labels:
                labels.append(label)
                label_count[label] = 1
            else:
                label_count[label] += 1
            index = list(labels).index(label)
            tags.append(index)

        labels = tags
        # ******************************** Create Arrays of zeros for each metric's data ***********************************
        
        _INCREMENT = 10
        print "Creating array of zeros..."

        # Max length of time series for each metric
        row_max_length = {'vmsram' : 0,
                  'tasks' : 0,
                  't_rscthnetno' : 0,
                  't_rscthhfsrb' : 0,
                  'c_ucpupct' : 0}

        # Calculate the aforementioned max length
        for data in data_set:
            for metric in _METRICS:
                try:
                    row_max_length[metric] = \
                        np.max([row_max_length[metric],
                                int((data_set[data][metric][-1][0] - data_set[data][metric][0][0]) / _INCREMENT) + 1])
                except:
                    continue

        data_matrices = {}

        # Store arrays of zeros with max length calculated above for each metric
        for metric in _METRICS:
            data_matrices[metric] = np.zeros(shape=(len(labels), row_max_length[metric]))

        # **************************************** Insert actual data into data_matrices ***********************************
        # The actual length of each time series for each metric

        print "Filling the arrays with actual data..."

        actual_row_lengths = {'vmsram' : [],
                  'tasks' : [],
                  't_rscthnetno' : [],
                  't_rscthhfsrb' : [],
                  'c_ucpupct' : []}

        # Iterate through the data and insert the actual data into the arrays of zeros to have arrays padded with zeros
        # where they contain no data
        for counter, data in enumerate(data_set):
            for metric in _METRICS:
                try:
                    data_for_metric = np.array(map(lambda x: [0, np.nan] if x==None else x, data_set[data][metric]))
                    length_of_ts = len(data_for_metric)
                    actual_row_lengths[metric].append(length_of_ts)
                    data_matrices[metric][counter][:length_of_ts] = [a[1] for a in data_for_metric]
                except:
                    actual_row_lengths[metric].append(0)

        # ****************************************** Normalize the  Data ***************************************************

        print "Starting normalization..."

        # Scale the data down to a [-1,1] interval
        for metric in _METRICS:
            maximum = np.nanmax(data_matrices[metric])
            minimum = np.nanmin(data_matrices[metric])
            for row_number, row in enumerate(data_matrices[metric]):
                if maximum > minimum:
                    row *= 2
                    row -= (maximum + minimum)
                    row /= (maximum - minimum)
                    for counter in range(actual_row_lengths[metric][row_number]):
                        if row[counter] > 1:
                            row[counter] = 1
                        if row[counter] < -1:
                            row[counter] = -1
                data_matrices[metric][row_number] = row
            print "max: ", np.nanmax(data_matrices[metric])
            print "min: ", np.nanmin(data_matrices[metric])
            print metric, " is normalized!"
        data_matrices['actual_lengths'] = actual_row_lengths
# ********************************** Image composition, Training, and Testing ******************************************
        phi_data = _create_phi_data(data_matrices)
        phi_test_data = {'actual_lengths': {}}
        phi_train_data = {'actual_lengths':{}}
        NUM_TEST_DATA = 200 
        for metric in _METRICS:
            phi_test_data[metric] = phi_data[metric][0:NUM_TEST_DATA]
            phi_train_data[metric] = phi_data[metric][NUM_TEST_DATA:]
            phi_test_data['actual_lengths'][metric] = phi_data['actual_lengths'][metric][0:NUM_TEST_DATA]
            phi_train_data['actual_lengths'][metric] = phi_data['actual_lengths'][metric][NUM_TEST_DATA:]
        test_labels = labels[0:NUM_TEST_DATA]
        train_labels = labels[NUM_TEST_DATA:]
# Matrices where element i contains the max length of the ith applications multiple metric time series
        max_dim_test = _compose_dimension_matrices(phi_test_data)
        max_dim_train = _compose_dimension_matrices(phi_train_data)

        nb_classes = len(np.unique(labels))
#        x_train, Y_train = _image_generator(phi_train_data, max_dim_train, train_labels, nb_classes)
#        x_test, Y_test = _image_generator(phi_test_data, max_dim_test, test_labels, nb_classes)
        
# Populate that list by composing temporary dictionaries and passing them through create_image and running it through the CNN
        NUM_EPOCHS = 100
        x = keras.layers.Input(shape=(None,None,1))
        conv1 = keras.layers.Conv2D(128, 8, 1, border_mode='same')(x)
        conv1 = keras.layers.normalization.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation('relu')(conv1)

        conv2 = keras.layers.Conv2D(256, 5, 1, border_mode='same')(conv1)
        conv2 = keras.layers.normalization.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

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
#        hist = model.fit(x_train, Y_train, validation_data=(x_test, Y_test), nb_epoch = NUM_EPOCHS, batch_size = 32, callbacks = [reduce_lr], verbose=1)
        hist = model.fit_generator(_image_generator(phi_train_data, max_dim_train, train_labels, nb_classes), len(train_labels), nb_epoch=NUM_EPOCHS,
                          verbose=1, validation_data=_image_generator(phi_test_data, max_dim_test, test_labels, nb_classes), validation_steps = len(test_labels), callbacks = [reduce_lr], use_multiprocessing=True, workers=20, max_queue_size=10)
        log = pd.DataFrame(hist.history)
        print log.loc[log['loss'].idxmin]['loss'], log.loc[log['loss'].idxmin]['val_acc']
       

if __name__ == "__main__":
    main()
