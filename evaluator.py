import tensorflow as tf

F = tf.app.flags.FLAGS


def evaluate():
    pass


if __name__ == '__main__':
    fl = tf.app.flags
    fl.DEFINE_string('config', "config.json", "config file path")
    fl.DEFINE_string('param', 'default', '')
    fl.DEFINE_boolean('parallel_exec', True, '')
    fl.DEFINE_string('save_dir', 'experiments/base_model', '')
    fl.DEFINE_string('data_dirs', './data/mnist|./data/', '')
    fl.DEFINE_string('data_files', None, '')
    fl.DEFINE_string('data_name', 'deepfashion', '')
    fl.DEFINE_string('data_mid_name', 'val', '')
    evaluate()
