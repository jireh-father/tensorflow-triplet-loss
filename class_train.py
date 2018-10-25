import os

import tensorflow as tf

from model import input_fn
from model import tfrecord_input_fn
from model.model_fn import model_fn
from model.utils import Params

fl = tf.app.flags
fl.DEFINE_string('config', "config.json", "config file path")
fl.DEFINE_string('param', 'default', '')
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
F = fl.FLAGS
if __name__ == '__main__':
    print(F.steps)

    F.steps = 5
    print(F.steps)
    import test

    test.a()

#   handle multi processing and gpu selecting
#   init config
#   auto create tfrecord
#
