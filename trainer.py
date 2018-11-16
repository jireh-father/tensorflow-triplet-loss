import os
import tensorflow as tf
from tensorflow.python.client import device_lib
from model import model_fn
import glob
import util
import time


def main(cf):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = F.gpu_no
    print("CUDA Visible device", device_lib.list_local_devices())
    # inputs_ph = tf.placeholder(tf.float32, [None, cf.input_size, cf.input_size, cf.input_channel],
    #                            name="inputs")
    # labels_ph = tf.placeholder(tf.int32, [None], name="labels")
    tf.set_random_seed(123)
    images_ph = tf.placeholder(tf.float32, [None, cf.input_size, cf.input_size, cf.input_channel], name="inputs")
    labels_ph = tf.placeholder(tf.int32, [None], name="labels")
    seed_ph = tf.placeholder(tf.int64, (), name="shuffle_seed")
    loss_op, train_op = model_fn.build_model(images_ph, labels_ph, cf, True)

    def train_pre_process(example_proto):
        features = {"image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
                    "image/class/label": tf.FixedLenFeature((), tf.int64, default_value=0),
                    'image/height': tf.FixedLenFeature((), tf.int64, default_value=0),
                    'image/width': tf.FixedLenFeature((), tf.int64, default_value=0)
                    }

        parsed_features = tf.parse_single_example(example_proto, features)
        image = parsed_features["image/encoded"]
        image = tf.image.decode_jpeg(parsed_features["image/encoded"], 3)
        image = tf.cast(image, tf.float32)

        image = tf.expand_dims(image, 0)
        image = tf.image.resize_image_with_pad(image, 224, 224)
        # image = tf.image.resize_bilinear(image, [224, 224], align_corners=False)
        image = tf.squeeze(image, [0])

        image = tf.divide(image, 255.0)
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)

        label = parsed_features["image/class/label"]

        return image, label

    files = glob.glob(os.path.join(cf.data_dir, "*_train*tfrecord"))
    files.sort()
    assert len(files) > 0
    num_examples = util.count_records(files)
    steps_each_epoch = int(num_examples / cf.batch_size)
    if num_examples % cf.batch_size > 0:
        steps_each_epoch += 1
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(train_pre_process)
    dataset = dataset.shuffle(cf.shuffle_buffer_size)
    dataset = dataset.repeat()
    dataset = dataset.batch(cf.sampling_buffer_size)
    dataset = dataset.prefetch(cf.prefetch_buffer_size)

    iterator = dataset.make_one_shot_iterator()
    # iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=cf.keep_checkpoint_max)
    if os.path.isdir(cf.save_dir):
        saver.restore(sess, tf.train.latest_checkpoint(cf.save_dir))

    epoch = 1
    steps = 1
    num_trained_images = 0
    while True:
        # sess.run(iterator.initializer, feed_dict={seed_ph: steps})
        try:
            start = time.time()
            tmp_images, tmp_labels = sess.run([images, labels])
            pair_indices = set()
            single_index_map = {}
            label_buffer = {}
            for i, tmp_label in enumerate(tmp_labels):
                if tmp_label in label_buffer:
                    pair_indices.add(i)
                    pair_indices.add(label_buffer[tmp_label])
                    if tmp_label in single_index_map:
                        del single_index_map[tmp_label]
                else:
                    label_buffer[tmp_label] = i
                    single_index_map[tmp_label] = i
            pair_indices = list(pair_indices)
            if len(pair_indices) > cf.batch_size:
                pair_indices = pair_indices[:cf.batch_size]
            elif len(pair_indices) < cf.batch_size:
                pair_indices += list(single_index_map.values())[:cf.batch_size - len(pair_indices)]
            # print(pair_indices)
            batch_images = tmp_images[pair_indices]
            batch_labels = tmp_labels[pair_indices]
            sampling_time = time.time() - start
            tmp_images = None
            tmp_labels = None
            start = time.time()
            loss, _ = sess.run([loss_op, train_op], feed_dict={images_ph: batch_images, labels_ph: batch_labels})
            train_time = time.time() - start
            print("[%d epoch(%d/%d), %d steps] sampling time: %f, train time: %f, loss: %f" % (
                epoch, steps % steps_each_epoch, steps_each_epoch, steps, sampling_time, train_time, loss))
            steps += 1
            num_trained_images += cf.batch_size

            if num_trained_images >= num_examples:
                saver.save(sess, cf.save_dir + "/model.ckpt", epoch)
                epoch += 1
                num_trained_images = 0
            if epoch >= cf.num_epochs:
                break
        except tf.errors.OutOfRangeError:
            break
    #
    # ds_list = []
    # label_map_list = []
    # label_cnt_list = []
    # dataset_list = cf.train_file_patterns.split("|")
    # for i, path_pattern in enumerate(dataset_list):
    #     tfrecord_files = glob.glob(path_pattern)
    #     assert len(tfrecord_files) > 0
    #     ds = dataset.build_dataset(tfrecord_files, cf.preprocessing_name, cf.shuffle_buffer_size,
    #                                cf.sampling_buffer_size, cf.num_map_parallel)
    #     ds_list.append(ds)
    #     label_map_list.append(util.create_label_map(tfrecord_files))
    #     label_cnt_list.append(len(label_map_list[i]))
    #     if i > 0:
    #         assert label_cnt_list[i] == label_cnt_list[i - 1]
    #         assert label_map_list[i] == label_map_list[i - 1]
    #
    # inputs_ph = tf.placeholder(tf.float32, [None, cf.input_size, cf.input_size, cf.input_channel],
    #                            name="inputs")
    # labels_ph = tf.placeholder(tf.int32, [None], name="labels")
    #
    # model_fn.build_model(inputs_ph, labels=labels_ph, mode=tf.estimator.ModeKeys.TRAIN)
    #
    # assert cf.model_name is not None
    # assert os.path.isfile(os.path.join("./models", cf.model_name + ".py"))
    # model_f = util.get_attr('models.%s' % cf.model_name, "build_model")
    #
    # data_list = model_f(inputs_ph, cf.embedding_size)
    #
    # assert cf.sampling_name is not None
    # assert os.path.isfile(os.path.join("./samplings", cf.sampling_name + ".py"))
    # sampling_f = util.get_attr('samplings.%s' % cf.sampling_name, "samplings")
    # data_list = sampling_f()
    #
    # sys.exit()
    #
    # ds = dataset.build_dataset(tfrecord_files, cf.preprocessing_name, cf.shuffle_buffer_size,
    #                            cf.sampling_buffer_size, cf.num_map_parallel)
    #
    # index_iterator = ds.make_initializable_iterator()
    #
    # img, index_labels = index_iterator.get_next()
    #
    # tf_config = tf.ConfigProto()
    # tf_config.gpu_options.allow_growth = True
    # sess = tf.Session()
    # from PIL import Image
    #
    # for i in range(2):
    #     sess.run(index_iterator.initializer)
    #     j = 0
    #     while True:
    #         try:
    #             images, ll = sess.run([img, index_labels])
    #             print(ll)
    #             im = Image.fromarray(images[0].astype('uint8'))
    #             im.show()
    #             # if j == 0:
    #             # im.save("%d.jpg" % i)
    #             break
    #             # j += 1
    #             # if i == 1:
    #             #     break
    #         except tf.errors.OutOfRangeError:
    #             break
    # sys.exit()


def train():
    pass


if __name__ == '__main__':
    fl = tf.app.flags

    fl.DEFINE_string('data_dir', 'D:\data\\fashion\image_retrieval\cafe24product\\tfrecord', '')
    fl.DEFINE_string('sampling_name', 'pk', 'pk, random...')
    fl.DEFINE_string('model_name', 'alexnet_v2', '')
    fl.DEFINE_integer('num_epochs', 10, '')
    fl.DEFINE_integer('input_size', 224, '')
    fl.DEFINE_integer('input_channel', 3, '')
    fl.DEFINE_integer('embedding_size', 128, '')
    fl.DEFINE_string('preprocessing_name', 'default_preprocessing', '')
    fl.DEFINE_integer('sampling_buffer_size', 2048, '')
    fl.DEFINE_integer('shuffle_buffer_size', 1000, '')
    fl.DEFINE_integer('prefetch_buffer_size', 2048, '')
    fl.DEFINE_string('save_dir', 'experiments/base_model', '')
    fl.DEFINE_string('triplet_strategy', 'batch_all', '')
    fl.DEFINE_float('margin', 0.5, '')
    fl.DEFINE_boolean('squared', False, '')
    fl.DEFINE_boolean('use_batch_norm', False, '')
    fl.DEFINE_string('data_mid_name', 'val', '')
    fl.DEFINE_integer('save_steps', 10000, '')
    fl.DEFINE_integer('save_epochs', 1, '')
    fl.DEFINE_integer('keep_checkpoint_max', 2, '')
    fl.DEFINE_integer('batch_size', 128, '')
    fl.DEFINE_integer('num_image_sampling', 4, '')
    fl.DEFINE_integer('num_single_image_max', 4, '')

    fl.DEFINE_integer('num_map_parallel', 4, '')
    fl.DEFINE_string('gpu_no', "0", '')
    fl.DEFINE_float('weight_decay', 0.00004, '')
    fl.DEFINE_string('optimizer', 'rmsprop', '"adadelta", "adagrad", "adam",''"ftrl", "momentum", "sgd"  "rmsprop".')
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
    fl.DEFINE_string('learning_rate_decay_type', 'exponential', '"fixed", "exponential",'' or "polynomial"')
    fl.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    fl.DEFINE_float('end_learning_rate', 0.0001, 'The minimal end learning rate used by a polynomial decay.')
    fl.DEFINE_float('label_smoothing', 0.0, 'The amount of label smoothing.')
    fl.DEFINE_float('learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
    fl.DEFINE_float('num_epochs_per_decay', 2.0, 'Number of epochs after which learning rate decays.')
    fl.DEFINE_float('moving_average_decay', 0.9999, 'The decay to use for the moving average.')

    F = fl.FLAGS
    main(F)
