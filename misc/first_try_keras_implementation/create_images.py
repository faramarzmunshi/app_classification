import numpy as np
import inception
from inception import transfer_values_cache
from keras.utils import np_utils
import pickle

def readucr(filename):
    data = np.loadtxt(filename, delimiter=',')
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y

def _save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def _process_strips(data, model):
    # Rescale between [0,255]
    data_min = min(data.min(axis=0))
    data -= data_min
    data_max = max(data.max(axis=0))
    data /= data_max
    data *= 255

    # Iterate through the dataset creating images of the strip and collect the outputs in cnn_output
    cnn_output = []
    for row in data:
        length = len(row)
        image = np.zeros([1,1,length,3])
        image[0,0,:,0] = row
        transfer_values = transfer_values_cache(model = model,
                                                images=image)
        cnn_output.append(transfer_values)

    return cnn_output

def _process_GADF_images(data, model):
    # Rescale between [-1,1]
    data_min = min(data.min(axis=1))
    data_max = max(data.max(axis=1))
    data *= 2
    data -= data_max + data_min
    data /= (data_max - data_min)

    # Arc cos the dataset
    data = np.arccos(data)

    # Iterate through the dataset and find the GADF and place it into an image to be processed by the CNN
    cnn_output = []
    for row_num, row in enumerate(data):
        length = len(row)
        tempsin = np.sin(row).reshape((length,1))
        tempcos = np.cos(row).reshape((length,1))
        new_row = np.dot(tempsin,tempcos.T) - np.dot(tempcos,tempsin.T)

        # Placeholder image
        image = np.zeros([1,length,length,3])

        # Fill with real data and rescale [0,255] for CNN
        image[0,:,:,0] = new_row
        image += 1
        image *= 127.5

        # Run through the CNN and append the final vector to cnn_output
        transfer_values = transfer_values_cache(model = model,
                                                images=image)
        cnn_output.append(transfer_values)

    return cnn_output

nb_epochs = 5000

_MODE = 'GADF'

flist = ['Adiac', 'Beef', 'CBF', 'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee', 'Cricket_X', 'Cricket_Y',
         'Cricket_Z',
         'DiatomSizeReduction', 'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR', '50words', 'FISH', 'Gun_Point',
         'Haptics',
         'InlineSkate', 'ItalyPowerDemand', 'Lighting2', 'Lighting7', 'MALLAT', 'MedicalImages', 'MoteStrain',
         'NonInvasiveFatalECG_Thorax1',
         'NonInvasiveFatalECG_Thorax2', 'OliveOil', 'OSULeaf', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII',
         'StarLightCurves', 'SwedishLeaf', 'Symbols',
         'synthetic_control', 'Trace', 'TwoLeadECG', 'Two_Patterns', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y',
         'uWaveGestureLibrary_Z', 'wafer', 'WordsSynonyms', 'yoga']

model = inception.Inception()

for each in flist:
    fname = each
    data_train, labels_train = readucr('UCR/' + fname + '/' + fname + '_TRAIN')
    data_test, labels_test = readucr('UCR/' + fname + '/' + fname + '_TEST')
    nb_classes = len(np.unique(labels_test))
    labels_train = (labels_train - labels_train.min()) / (labels_train.max() - labels_train.min()) * (nb_classes - 1)
    labels_test = (labels_test - labels_test.min()) / (labels_test.max() - labels_test.min()) * (nb_classes - 1)

    hot_labels_train = np_utils.to_categorical(labels_train, nb_classes)
    hot_labels_test = np_utils.to_categorical(labels_test, nb_classes)

    # Z Normalization
    data_train_mean = data_train.mean()
    data_train_std = data_train.std()
    data_train = (data_train - data_train_mean) / (data_train_std)
    data_test = (data_test - data_train_mean) / (data_train_std)

    if _MODE == 'strip':
        train_cnn_vectors = _process_strips(data_train, model)
        test_cnn_vectors = _process_strips(data_test, model)
        directory = 'strip_images/'
    elif _MODE == 'GADF':
        train_cnn_vectors = _process_GADF_images(data_train, model)
        test_cnn_vectors = _process_GADF_images(data_test, model)
        directory = 'GADF_images/'

    _FILENAME = "{0}{1}_{2}_CNNOUT".format(directory, fname, _MODE)

    for i, label in enumerate(labels_train):
        train_cnn_vectors[i] = [label] + train_cnn_vectors[i].tolist()
    for i, label in enumerate(labels_test):
        test_cnn_vectors[i] = [label] + test_cnn_vectors[i].tolist()

    _save_object(train_cnn_vectors, _FILENAME + '_TRAIN')
    _save_object(test_cnn_vectors, _FILENAME + '_TEST')
