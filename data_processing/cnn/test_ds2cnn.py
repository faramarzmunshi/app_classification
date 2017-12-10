
# LOADS THE DATASET FORMAT FROM JSON_TO_DATASCIENCE_FORMAT.PY AND RUNS THE DATASETS IN RLIST THROUGH THE CNN, PRODUCING
# AND SAVING THOSE VECTORS

# *********************************************** Imports **************************************************************
import numpy as np
# Needed for loading objects
import pickle
# Functions and classes for loading and using the Inception model.
import inception
# Import a helper-function for caching the transfer-values of the Inception model
from inception import transfer_values_cache
from pprint import pprint
from datetime import datetime
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
# *********************************************** Functions ************************************************************
def _save_obj(obj, name):
    """
    Saves an object in a pickle file.
    :param obj: object to be saved
    :param name: name of file/file path
    """
    with open('/bigdisk/pickles/' + name, 'w') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def _load_obj(name):
    """
    Load a pickled object
    :param name: name of pickled object/file location
    :return: @rtype = obj - unpickled object
    """
    with open('/bigdisk/pickles/' + name, 'r') as f:
        return pickle.load(f)

def _create_phi_data(training_data, test_data):
    """
    Creates the phi version of the training and test datasets.
    :param training_data: set of training data for dataset
    :param test_data: set of test data for dataset
    :return: @rtype = dict - the phi version of the training and test datasets
    """
    _METRICS = ['vmsram', 'tasks', 't_rscthnetno', 't_rscthhfsrb', 'c_ucpupct']
    phi_training_data = {}
    phi_test_data = {}

    # Iterate and compute arccos of each time series in training and test data
    for key in training_data.keys():
        if key in _METRICS:
            phi_training_data[key] = np.arccos(training_data[key])
            phi_test_data[key] = np.arccos(test_data[key])
        else:
            phi_training_data[key] = training_data[key]
            phi_test_data[key] = test_data[key]

    return phi_training_data, phi_test_data

def _compose_dimension_matrices(training_data, test_data):
    """
    Composes dimension matrices that are needed for smartly batching
    :param training_data: set of training data
    :param test_data: set of test data
    :return: @rtype = np.array - arrays of max length for each time series in both test and training data
    """
    # Number of the training data
    _NUM_TRAINING_DATA = len(training_data[_METRICS[0]])

    # Number of the test data
    _NUM_TEST_DATA = len(test_data[_METRICS[0]])

    # Dictionary that contains dimensions for each metric for each time series
    dim_matrix = {}

    # The largest length of time series for all metrics in a certain row in the training and test data
    max_dim_train = []
    max_dim_test = []

    # Filling in the dimension matrix with the max length across metrics for a single application
    for row in range(_NUM_TRAINING_DATA):
        for metric in _METRICS:
            dim = training_data['actual_lengths'][metric][row]
            if dim != 0:
                dim_matrix[metric] = training_data['actual_lengths'][metric][row]
            else:
                dim_matrix[metric] = 1
        max_dim_train.append(max(dim_matrix.values()))

    for row in range(_NUM_TEST_DATA):
        for metric in _METRICS:
            dim = test_data['actual_lengths'][metric][row]
            if dim != 0:
                dim_matrix[metric] = test_data['actual_lengths'][metric][row]
            else:
                dim_matrix[metric] = 1
        max_dim_test.append(max(dim_matrix.values()))

    return max_dim_train, max_dim_test

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
    images = np.zeros(shape=(len(list_of_dicts), len(_METRICS) * largest_dim, largest_dim, 3))

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

    print "Creating the images took {0}".format(datetime.now() - timer)
    return images

# ***************************************** Loading model and variables ************************************************

def main():
	# Realms to be processed through the CNN
	# NOTE: THESE MUST HAVE BEEN THROUGH JSON_TO_DATASCIENCE_FORMAT.py FIRST!
	# The well-named realms we've determined were this list
	#REALMS = ['dataminesjc', 'nbcuni-centralperk2', 'rubicon-fmap', 'gamut-prod', 'rubicon-fbmp', 'nbcuni-superstore', 'jumpshot-jsc', 'rubicon-fbmq', 'rally-health-integration', 'prudential-nj-exp2', 'rubicon-fmaq']
	REALMS = ['gamut-prod']

	# Iterates through all realms
	for REALM in REALMS:
		# Metrics that we have defined
		_NUM_METRICS = len(_METRICS)

		# Load the training/test data as well as their respective lengths
		training_data = _load_obj(REALM + '^training_data')
		training_data['actual_lengths'] = _load_obj(REALM + '^training_length')
		test_data = _load_obj(REALM + '^test_data')
		test_data['actual_lengths'] = _load_obj(REALM + '^test_length')
                print "Loaded data"

		# Load the labels
		labels = _load_obj(REALM + '^labels')
		_NUM_LABELS = len(labels)

		# The number of data points in training and test so we can iterate over them later
		max_dim_train, max_dim_test = _compose_dimension_matrices(training_data, test_data)
                new_training_data = []
                for row, longest_metric_length in enumerate(max_dim_train):
                    concatted = np.array([])
                    print "row, longest_metric_length: ", row, longest_metric_length
                    for metric in _METRICS:
                        filler = np.zeros(longest_metric_length)
                        actual_length = training_data['actual_lengths'][metric][row]
    			print "actual_length: ", actual_length
                        filler[0:actual_length] = training_data[metric][row][0:actual_length]
                        concatted = np.concatenate((concatted, filler), axis=0)
                    new_training_data.append(concatted.tolist())
		print "Created long training data"
                
                new_test_data = []
                for row, longest_metric_length in enumerate(max_dim_test):
                    concatted = np.array([])
                    for metric in _METRICS:
                        filler = np.zeros(longest_metric_length)
                        actual_length = test_data['actual_lengths'][metric][row]
                        filler[0:actual_length] = training_data[metric][row][0:actual_length]
                        concatted = np.concatenate((concatted,filler), axis=0)
                    new_test_data.append(concatted.tolist())
 		print "Created long test data"
		
                phi_training_data = [np.arccos(row) for row in new_training_data]
                phi_test_data = [np.arccos(row) for row in new_test_data]
		print "Phi'd the data"
		
                model = inception.Inception()
		print "Loaded the model"
                
                test_vectors = []
                for row in phi_test_data:
                    temp_sin = np.sin(row)
                    temp_sin = temp_sin.reshape((len(temp_sin),1))
                    temp_cos = np.cos(row)
                    temp_cos = temp_cos.reshape((len(temp_cos),1))
                    image = np.zeros((1,len(temp_sin),len(temp_sin),3))
                    image[0,:,:,0] = _interpolation(np.dot(temp_sin,temp_cos.T) - np.dot(temp_cos,temp_sin.T))
                    image+=1
                    image*=127.5
                    output_vector = transfer_values_cache(images=image, model=model)
                    test_vectors.append(output_vector)
		print "Done with test cnn"
                phi_test_data = None
		# Saves the objects relevant to the DNN next step for test
		_save_obj(test_vectors, REALM + '^cnn_test_output_concat')
		print "Saved test cnn"

		# All the same stuff as above, but for training data
                train_vectors = []
                for row in phi_training_data:
                    temp_sin = np.sin(row)
                    temp_sin = temp_sin.reshape((len(temp_sin),1))
                    temp_cos = np.cos(row)
                    temp_cos = temp_cos.reshape((len(temp_cos),1))
                    image = np.zeros((1,len(temp_sin),len(temp_sin),3))
                    image[0,:,:,0] = _interpolation(np.dot(temp_sin,temp_cos.T) - np.dot(temp_cos,temp_sin.T))
                    image+=1
                    image*=127.5
                    output_vector = transfer_values_cache(images=image, model=model)
                    train_vectors.append(output_vector)
		print "Done with train cnn"
                phi_training_data = None
		_save_obj(train_vectors, REALM + '^cnn_training_output_concat')
		print "Saved train cnn"

if __name__ == '__main__':
        _METRICS = ['vmsram', 'tasks', 't_rscthnetno', 't_rscthhfsrb', 'c_ucpupct']
	main()











































