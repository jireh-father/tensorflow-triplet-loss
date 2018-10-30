import h5py
import numpy as np

BATCH_SIZE = 32


def get_batch(file_h5, features, batch_number, batch_size=32):
    """Get a batch of the dataset

    Args:
        file_h5(str): path of the dataset
        features(list(str)): list of names of features present in the dataset
            that should be returned.
        batch_number(int): the id of the batch to be returned.
        batch_size(int): the mini-batch size
    Returns:
        A list of numpy arrays of the requested features"""
    list_of_arrays = []
    lb, ub = batch_number * batch_size, (batch_number + 1) * batch_size
    for feature in features:
        list_of_arrays.append(file_h5[feature][lb: ub])
    return list_of_arrays


# open the file
file_h5 = h5py.File('ssense_256_256_train.h5', mode='r')
# define the features to be retrieved
list_of_features = ['input_image', 'input_description']

dataset_len = len(file_h5['input_image'])
nb_batches = int(dataset_len / BATCH_SIZE)

batch_nb = np.random.randint(0, nb_batches)

# get the first batch of the data
list_of_arrays = get_batch(file_h5, list_of_features, batch_nb, BATCH_SIZE)
# close the file
file_h5.close()
