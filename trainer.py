import os
import util
import tensorflow as tf
from tensorflow.python.client import device_lib
from core import model_fn
import glob
import dataset


def main(config):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = F.gpu_no
    print("CUDA Visible device", device_lib.list_local_devices())

    ds_list = []
    label_map_list = []
    label_cnt_list = []
    dataset_list = config.train_file_patterns.split("|")
    for i, path_pattern in enumerate(dataset_list):
        tfrecord_files = glob.glob(path_pattern)
        assert len(tfrecord_files) > 0
        ds = dataset.build_dataset(tfrecord_files, config.preprocessing_name, config.shuffle_buffer_size,
                                   config.sampling_buffer_size, config.num_map_parallel)
        ds_list.append(ds)
        label_map_list.append(util.create_label_map(tfrecord_files))
        label_cnt_list.append(len(label_map_list[i]))
        if i > 0:
            assert label_cnt_list[i] == label_cnt_list[i - 1]
            assert label_map_list[i] == label_map_list[i - 1]

    inputs_ph = tf.placeholder(tf.float32, [None, config.input_size, config.input_size, config.input_channel],
                               name="inputs")
    labels_ph = tf.placeholder(tf.int32, [None], name="labels")

    model_fn.build_model(inputs_ph, labels=labels_ph, mode=tf.estimator.ModeKeys.TRAIN)

    assert config.model_name is not None
    assert os.path.isfile(os.path.join("./models", config.model_name + ".py"))
    model_f = util.get_attr('models.%s' % config.model_name, "build_model")

    data_list = model_f(inputs_ph, config.embedding_size)

    assert config.sampling_name is not None
    assert os.path.isfile(os.path.join("./samplings", config.sampling_name + ".py"))
    sampling_f = util.get_attr('samplings.%s' % config.sampling_name, "samplings")
    data_list = sampling_f()

    sys.exit()

    ds = dataset.build_dataset(tfrecord_files, config.preprocessing_name, config.shuffle_buffer_size,
                               config.sampling_buffer_size, config.num_map_parallel)

    index_iterator = ds.make_initializable_iterator()

    img, index_labels = index_iterator.get_next()

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session()
    from PIL import Image

    for i in range(2):
        sess.run(index_iterator.initializer)
        j = 0
        while True:
            try:
                images, ll = sess.run([img, index_labels])
                print(ll)
                im = Image.fromarray(images[0].astype('uint8'))
                im.show()
                # if j == 0:
                # im.save("%d.jpg" % i)
                break
                # j += 1
                # if i == 1:
                #     break
            except tf.errors.OutOfRangeError:
                break
    sys.exit()


def train():
    pass


if __name__ == '__main__':
    fl = tf.app.flags

    fl.DEFINE_string('train_file_patterns',
                     'E:/data/adience_kaggle/test/*.tfrecord|E:/data/adience_kaggle/test2/*.tfrecord', '')
    fl.DEFINE_boolean('parallel_exec', True, '')
    fl.DEFINE_string('sampling_name', 'pk', 'pk, random...')
    fl.DEFINE_string('model_name', 'alexnet_v2', '')
    fl.DEFINE_integer('nums_sampling_classes', 4, 'ony in case of pk sampling.')
    fl.DEFINE_integer('input_size', 224, '')
    fl.DEFINE_integer('input_channel', 3, '')
    fl.DEFINE_integer('embedding_size', 128, '')
    fl.DEFINE_string('preprocessing_name', 'default_preprocessing', '')
    fl.DEFINE_integer('shuffle_buffer_size', 1000, '')
    fl.DEFINE_integer('sampling_buffer_size', 1024, '')
    fl.DEFINE_integer('num_map_parallel', 4, '')
    fl.DEFINE_string('gpu_no', "0", '')
    fl.DEFINE_float('weight_decay', 0.00004, '')
    fl.DEFINE_string('optimizer', 'rmsprop', 'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
                                             '"ftrl", "momentum", "sgd"  "rmsprop".')
    fl.DEFINE_float('adadelta_rho', 0.95, 'The decay rate for adadelta.')
    fl.DEFINE_float('adagrad_initial_accumulator_value', 0.1, 'Starting value for the AdaGrad accumulators.')
    fl.DEFINE_float('adam_beta1', 0.9, 'The exponential decay rate for the 1st moment estimates.')
    fl.DEFINE_float('adam_beta2', 0.999, 'The exponential decay rate for the 2nd moment estimates.')
    fl.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')
    fl.DEFINE_float('ftrl_learning_rate_power', -0.5, 'The learning rate power.')
    fl.DEFINE_float('ftrl_initial_accumulator_value', 0.1, 'Starting value for the FTRL accumulators.')
    fl.DEFINE_float('ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')
    fl.DEFINE_float('ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')
    fl.DEFINE_float('momentum', 0.9, 'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
    fl.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
    fl.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

    #######################
    # Learning Rate Flags #
    #######################

    fl.DEFINE_string('learning_rate_decay_type', 'exponential',
                     'Specifies how the learning rate is decayed. One of "fixed", "exponential",'' or "polynomial"')
    fl.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    fl.DEFINE_float('end_learning_rate', 0.0001,
                    'The minimal end learning rate used by a polynomial decay learning rate.')
    fl.DEFINE_float('label_smoothing', 0.0, 'The amount of label smoothing.')
    fl.DEFINE_float('learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

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
    main(F)
