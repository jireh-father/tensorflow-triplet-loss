"""Create the input data pipeline using `tf.data`"""

from model import tfrecords_dataset as td


def train_input_fn(data_dir, params):
    """Train input function for the MNIST dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = td.train(data_dir)
    dataset = dataset.shuffle(1000)  # whole dataset into the buffer
    dataset = dataset.repeat(params.num_epochs)  # repeat for multiple epochs
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(params.batch_size)  # make sure you always have one batch ready to serve
    return dataset


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


def query_label_fn(data_dir, params):
    """Test input function for the MNIST dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = td.query_label(data_dir)
    dataset = dataset.batch(params.query_cnt)
    # dataset = dataset.prefetch(params.batch_size)  # make sure you always have one batch ready to serve
    return dataset


def index_label_fn(data_dir, params):
    """Test input function for the MNIST dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = td.index_label(data_dir)
    dataset = dataset.batch(params.index_cnt)
    # dataset = dataset.prefetch(params.batch_size)  # make sure you always have one batch ready to serve
    return dataset
