import os
import util
import tensorflow as tf
from tensorflow.python.client import device_lib


def main(config):
    print("CUDA Visible device", device_lib.list_local_devices())

    assert config.sampling_name is not None
    assert os.path.isfile(os.path.join("./samplings", config.sampling_name + ".py"))
    sampling_f = util.get_attr('samplings.%s' % config.sampling_name, "samplings")
    data_list = sampling_f()

    assert config.model_name is not None
    assert os.path.isfile(os.path.join("./models", config.model_name + ".py"))
    sampling_f = util.get_attr('models.%s' % config.model_name, "build_model")
    data_list = sampling_f()


def train():
    pass


if __name__ == '__main__':
    fl = tf.app.flags
    fl.DEFINE_string('train_consumer_path_patterns', './dataset/*.tfrecord|./dataset2/*.tfrecord', '')
    fl.DEFINE_string('train_shop_path_patterns', './dataset/*.tfrecord|./dataset2/*.tfrecord', '')
    fl.DEFINE_boolean('parallel_exec', True, '')
    fl.DEFINE_string('sampling_name', 'pk', 'pk, random...')
    fl.DEFINE_string('model_name', 'alexnet_v2', '')
    fl.DEFINE_integer('nums_sampling_classes', 4, 'ony in case of pk sampling.')

    fl.DEFINE_string('save_dir', 'experiments/base_model', '')
    fl.DEFINE_string('data_dirs', './data/mnist|./data/', '')
    fl.DEFINE_string('data_files', None, '')
    fl.DEFINE_string('data_name', 'deepfashion', '')
    fl.DEFINE_string('data_mid_name', 'val', '')
    fl.DEFINE_integer('test_step', 10000, '')
    fl.DEFINE_integer('save_step', 10000, '')
    fl.DEFINE_integer('save_epoch', 10000, '')
    fl.DEFINE_integer('save_max', 10000, '')
    fl.DEFINE_integer('epochs', 10000, '')
    fl.DEFINE_integer('steps', 10000, '')
    fl.DEFINE_boolean('random_shuffling', True, '')
    fl.DEFINE_boolean('use_create_dataset_phase', True, '')
    fl.DEFINE_boolean('use_train_phase', True, '')
    fl.DEFINE_boolean('use_val_phase', False, '')
    fl.DEFINE_boolean('use_test_phase', True, '')
    F = tf.app.flags.FLAGS
    main(F)
