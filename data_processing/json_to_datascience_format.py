
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
import re
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
        out[i, label_vector[i]] = 1
    return out

def _save_obj(obj, name):
    """
    Saves an object in a pickle file.
    :param obj: object to be saved
    :param name: name of file/file path
    """
    directory = '/bigdisk/pickles/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(directory + name, 'w') as outfile:
    	pickle.dump(obj, outfile)

def _compile_filepath_list(input):
        import glob, os
        filepaths = []
        for root, dirs, files in os.walk('./cnn/data/newjsons/'):
            for file in files:
                if file.endswith(input):
                    filepath = os.path.join(root,file)
                    filepaths += [filepath]
        return filepaths

# ************************************* Constants and Variable Loading *************************************************

def main():
    # Name of file to load data from
    filenames = _compile_filepath_list('.json')
    for filename in filenames:
        print "Opened and reading {0}...".format(filename)
        _FILE_NAME = filename

        _CLUSTER = _FILE_NAME.split('_')[0].split('/')[-1]

        # List of all metrics collected for each job
        _METRICS = ['vmsram', 'tasks', 't_rscthnetno', 't_rscthhfsrb', 'c_ucpupct']

        # Time between data points
        _INCREMENT = 10

        # Cutoff for number of characters to define which labels are the same
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
            changed_name = re.sub('[0-9][0-9][0-9]+', '', name)
            changed_name = re.sub('__.*', '', changed_name)
            label = re.sub('_.*', '', changed_name)
            label = re.sub('-[0-9]+', '', label)
            label = re.sub('[0-9]+-', '', label)            
            if label not in labels:
                labels.append(label)
                label_count[label] = 1
            else:
                label_count[label] += 1
            index = list(labels).index(label)
            tags.append(index)

        # ******************************** Create Arrays of zeros for each metric's data ***********************************

        print "Creating array of zeros..."

        _NUM_LABELS = len(labels)

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
            data_matrices[metric] = np.zeros(shape=(int(np.sum(label_count.values())), row_max_length[metric]))

        # Store labels in data_matrices so all data is in one place
        data_matrices['labels'] = tags

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
            print np.nanmax(data_matrices[metric])
            print np.nanmin(data_matrices[metric]) 
            print metric, " is normalized!"

        # ************************************** Constitute the Training and Test Data *************************************

        # Number of total elements in the dataset
        _NUM_ELEMENTS = data_matrices['vmsram'].shape[0]

        # Percentage of data used for training data
        _TRAINING_PERCENTAGE = .90

        # Training data for each metric and respective lengths
        training_data = {}
        training_length = {}

        # Test data for each metric and respective lengths
        test_data = {}
        test_length = {}

        # Randomly chosen indices of rows to constitute the training data
        train_indices = np.random.choice(range(_NUM_ELEMENTS),
                                         int(_NUM_ELEMENTS * _TRAINING_PERCENTAGE),
                                         replace=False)

        # The rest of the row indices to constitute the test data
        test_indices = list(set(range(_NUM_ELEMENTS)) - set(train_indices))

        # Fill the above dictionaries with their respective data
        for metric in _METRICS:
            training_data[metric] = list(data_matrices[metric][train_indices, :])
	    data_matrices[metric][train_indices, :] = None
            training_length[metric] = list(np.asarray(actual_row_lengths[metric])[train_indices])
            test_data[metric] = list(data_matrices[metric][test_indices, :])
	    data_matrices[metric][test_indices, :] = None
            test_length[metric] = list(np.asarray(actual_row_lengths[metric])[test_indices])

        # Store labels and their one-hot encoded versions
        training_data['labels'] = list(np.asarray(data_matrices['labels'])[train_indices])
        training_data['hot_labels'] = list(_one_hot_encode(training_data['labels'], _NUM_LABELS))

        test_data['labels'] = list(np.asarray(data_matrices['labels'])[test_indices])
        test_data['hot_labels'] = list(_one_hot_encode(test_data['labels'], _NUM_LABELS))

        # Save dictionaries to files
        _save_obj(training_data, _CLUSTER + "^training_data")
	training_data = None
        _save_obj(training_length, _CLUSTER + "^training_length")
	training_length = None
        _save_obj(test_data, _CLUSTER + "^test_data")
	test_data = None
        _save_obj(test_length, _CLUSTER + "^test_length")
	test_length = None
        _save_obj(labels, _CLUSTER + "^labels")
	labels = None

if __name__ == "__main__":
    main()
