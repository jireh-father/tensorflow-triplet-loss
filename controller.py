import os, json
import util, trainer
import tensorflow as tf

from multiprocessing import Process
from model import input_fn
from model import tfrecord_input_fn
from model.model_fn import model_fn
from model.utils import Params

fl = tf.app.flags
fl.DEFINE_string('config', "config.json", "config file path")

fl.DEFINE_string('param', 'default', '')
fl.DEFINE_boolean('parallel_exec', True, '')
fl.DEFINE_string('save_dir', 'experiments/base_model', '')

fl.DEFINE_string('data_dirs', './data/mnist|./data/', '')
fl.DEFINE_string('train_files', None, '')
fl.DEFINE_string('train_val_files', None, '')
fl.DEFINE_string('train_test_files', None, '')

fl.DEFINE_string('data_name', 'deepfashion', '')
fl.DEFINE_string('data_mid_name', 'val', '')
fl.DEFINE_integer('test_step', 10000, '')
fl.DEFINE_integer('save_step', 10000, '')
fl.DEFINE_integer('save_epoch', 10000, '')
fl.DEFINE_integer('save_max', 10000, '')
fl.DEFINE_integer('epochs', 10000, '')
fl.DEFINE_integer('steps', 10000, '')
fl.DEFINE_string('gpu_no', "0", '')
fl.DEFINE_boolean('random_shuffling', True, '')

fl.DEFINE_boolean('use_train_phase', True, '')
fl.DEFINE_boolean('use_val_phase', False, '')
fl.DEFINE_boolean('use_test_phase', True, '')
F = fl.FLAGS


def runner():
    p = Process(target=trainer.main, args=(F,))
    p.daemon = True
    p.start()
    if not F.parallel_exec:
        p.join()


if __name__ == '__main__':
    assert F.config is not None
    assert os.path.isfile(F.config)
    configs = json.load(open(F.config))
    if len(configs) < 1:
        runner()
    else:
        for config in configs:
            bak_conf = util.set_flags(config)
            runner()
            util.restore_flags(bak_conf)
