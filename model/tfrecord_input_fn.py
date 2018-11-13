"""Create the input data pipeline using `tf.data`"""

from model import tfrecords_dataset as td
import tensorflow as tf


def train_input_fn(data_dir, params):
    """Train input function for the MNIST dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = td.train(data_dir)
    # if hasattr(params, "shuffle_rand_seed"):
    #     shuffle_rand_seed = params.shuffle_rand_seed
    # else:
    #     shuffle_rand_seed = 1
    # import tensorflow as tf
    # shuffle_rand_seed_ph = tf.placeholder(tf.int64, ())
    dataset = dataset.shuffle(1000)  # whole dataset into the buffer
    dataset = dataset.repeat(
        params.num_epochs)  # r                                                                                                                                                                                                                                                                                                                 epeat for multiple epochs
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(params.batch_size)  # make sure you always have one batch ready to serve
    return dataset  # , shuffle_rand_seed_ph


def train_input_fn_once(data_dir, params):
    """Train input function for the MNIST dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = td.train(data_dir)
    dataset = dataset.batch(params.batch_size)
    return dataset


def test_input_fn(data_dir, params):
    """Test input function for the MNIST dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = td.test(data_dir)
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(params.batch_size)  # make sure you always have one batch ready to serve
    return dataset


def query_input_fn(data_dir, params):
    """Test input function for the MNIST dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = td.query(data_dir)
    dataset = dataset.batch(params.batch_size)
    # dataset = dataset.prefetch(params.batch_size)  # make sure you always have one batch ready to serve
    return dataset


def index_input_fn(data_dir, params):
    """Test input function for the MNIST dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = td.index(data_dir)
    dataset = dataset.batch(params.batch_size)
    # dataset = dataset.prefetch(params.batch_size)  # make sure you always have one batch ready to serve
    return dataset


def train_label_fn(data_dir, params):
    """Test input function for the MNIST dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = td.train_label(data_dir)
    dataset = dataset.batch(params.train_size)
    # dataset = dataset.prefetch(params.batch_size)  # make sure you always have one batch ready to serve
    return dataset


def test_label_fn(data_dir, params):
    """Test input function for the MNIST dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = td.test_label(data_dir)
    dataset = dataset.batch(params.eval_size)
    # dataset = dataset.prefetch(params.batch_size)  # make sure you always have one batch ready to serve
    return dataset


def count_records(tfrecord_filenames):
    c = 0
    for fn in tfrecord_filenames:
        for _ in tf.python_io.tf_record_iterator(fn):
            c += 1
    return c


def query_label_fn(data_dir, params):
    """Test input function for the MNIST dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset, files = td.query_label(data_dir)
    cnt = count_records(files)
    dataset = dataset.batch(cnt)
    # dataset = dataset.prefetch(params.batch_size)  # make sure you always have one batch ready to serve
    return dataset, cnt


def index_label_fn(data_dir, params):
    """Test input function for the MNIST dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset, files = td.index_label(data_dir)
    cnt = count_records(files)
    dataset = dataset.batch(cnt)
    # dataset = dataset.prefetch(params.batch_size)  # make sure you always have one batch ready to serve
    return dataset, cnt
