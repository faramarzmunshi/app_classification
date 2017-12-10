
# FOR LOADING PULLED DATA FOR A CLUSTER FROM JSON FORMAT AND TESTING IT ON THE FINAL MODEL

# *********************************************** Imports **************************************************************
import json
from PIL import Image
from matplotlib import pyplot as plt
# for parsing raw data
import pickle
# for saving data to files
import numpy as np
# for splitting train and test data
import os
# for traversing file paths
# Functions and classes for loading and using the Inception model.
import inception
# Import a helper-function for caching the transfer-values of the Inception model
from inception import transfer_values_cache
from pprint import pprint
from datetime import datetime
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
_METRICS = ['vmsram', 'tasks', 't_rscthnetno', 't_rscthhfsrb', 'c_ucpupct']

# *********************************************** Functions ************************************************************
def create_label_vector(time_series, max_y):
    """
    Create a label vector from a time series.
    :param time_series: the input time series @type=list of tuples
    :return: label vector @type=list
    """
    timer = datetime.now()
    NUM_DIVS_WIDTH = 2
    NUM_DIVS_HEIGHT = 3

    # Decompose time series into x and y coordinates
    y = time_series
    num_points = len(time_series)
    x = range(num_points)
#    print x, len(x)
#    print y, len(y)

    if num_points == 0:
        return (['N']*NUM_DIVS_WIDTH).append(num_points)   

    # Set up variables
    TOTAL_HEIGHT = max_y
    num_pixels_in_div_height = TOTAL_HEIGHT * 1.0 / NUM_DIVS_HEIGHT
 
    TOTAL_WIDTH = max(x) - min(x)
    num_pixels_in_div_width = TOTAL_WIDTH * 1.0 / NUM_DIVS_WIDTH    

    length = TOTAL_WIDTH
    
    compressed = np.zeros((NUM_DIVS_HEIGHT, NUM_DIVS_WIDTH))

    for div_num_height in range(0, NUM_DIVS_HEIGHT):
        for div_num_width in range(0, NUM_DIVS_WIDTH):
            beg_x = div_num_width * num_pixels_in_div_width
            end_x = (div_num_width + 1) * num_pixels_in_div_width
            beg_y = div_num_height * num_pixels_in_div_height
            end_y = (div_num_height + 1) * num_pixels_in_div_height
            for x_el, y_el in zip(x,y):
                if x_el >= beg_x and x_el <= end_x and y_el >= beg_x and y_el <= end_y:
		    compressed[div_num_height][div_num_width] = 1
                    break
    
#    print compressed
        
    # Categories for final label
    categories = {0: 'L',
                  1: 'M',
                  2: 'H'}

    # Iterate through the rows starting from the last row and create final label
    label = ['N']*NUM_DIVS_WIDTH
    for category, row in enumerate(compressed[::-1]):
        for index, el in enumerate(row):
            if el:
                label[index] = categories[category]
    
#    print "This took: ", datetime.now()-timer
    return label[::-1]

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

# ************************************* Constants and Variable Loading *************************************************

def main():
    # Name of file to load data from
    filenames = ['./data/newjsons/rubicon-fmap_1505242800000_7_days_app_data.json']
    # Can also point to a directory and look at all files ending in a certain file extension
    # filenames = _compile_filepath_list('.json')
    for filename in filenames:
        print "Opened and reading {0}...".format(filename)
        _FILE_NAME = filename

        REALM = _FILE_NAME.split('_')[0].split('/')[-1]

        # List of all metrics collected for each job
        _METRICS = ['vmsram', 'tasks', 't_rscthnetno', 't_rscthhfsrb', 'c_ucpupct']

        # Open data file and load data
        with open(_FILE_NAME) as infile:
            data_set = json.load(infile)

        # ************************************* Constitute the label vector ************************************************
        app_ids = []
        # Iterate through the data set and identify unique labels by the first _NUM_CHAR characters in the name of the jobs
        for app_id in data_set:
	    name = data_set[app_id]['job_name']
            lengths = []
            for metric in _METRICS:
                try:
		    lengths.append(len(data_set[app_id][metric]))
		except:
		    continue
            length = max(lengths)
	    app_ids.append((app_id, name, length))
            
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
            data_matrices[metric] = np.zeros(shape=(len(app_ids), row_max_length[metric]))

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
	
#        maximums = {}
#        for metric in _METRICS:
#            maximums[metric] = np.nanmax(data_matrices[metric])
# 	print maximums
#        for row_num in range(len(app_ids)):
#            final_label = []
#            for metric in _METRICS:
#                l = create_label_vector(data_matrices[metric][row_num][0:actual_row_lengths[metric][row_num]], maximums[metric])
#                final_label.append(l)
#            print final_label
#            app_ids[row_num] = (app_ids[row_num][0], app_ids[row_num][1], final_label)
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
            print metric, " is normalized!"
        
        data_matrices['actual_lengths'] = actual_row_lengths
        model = inception.Inception()
	# ********************************** Image composition, Training, and Testing ******************************************
        _NUM_DATA = data_matrices[_METRICS[0]].shape[0]
        phi_data = _create_phi_data(data_matrices) 
        # Matrices where element i contains the max length of the ith applications multiple metric time series
        max_dim = _compose_dimension_matrices(phi_data)

	# List of all CNN outputs to every image in the test dataset
        cnn_vectors = []

	# Populate that list by composing temporary dictionaries and passing them through create_image and running it through the CNN
        for row in range(_NUM_DATA):
            temp_dict = {'actual_lengths': {}}
            for metric in _METRICS:
                temp_dict[metric] = phi_data[metric][row]
                temp_dict['actual_lengths'][metric] = phi_data['actual_lengths'][metric][row]
            input_dict = [temp_dict]
            largest_dim = max_dim[row]
            image = _create_image(input_dict, largest_dim)

            output_vector = transfer_values_cache(images=image,
                                                  model=model)
            cnn_vectors.append(output_vector)
        # Save the output of the CNN to cnn_vectors and the application ids with their corresponding names in numerical order to app_ids
	# Meaning the first vector in cnn_vectors corresponds to the first app_id with its name
        _save_obj(cnn_vectors, REALM + '^cnn_vectors')
        _save_obj(app_ids, REALM + '^app_ids')        

if __name__ == "__main__":
    main()
