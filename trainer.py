import os
import tensorflow as tf
from tensorflow.python.client import device_lib
from model import model_fn
import glob
import util
from datetime import datetime
import time
import numpy as np
from numba import cuda
from preprocessing import preprocessing_factory
import socket
import traceback

slim = tf.contrib.slim
notify_params = [
    "gpu_no",
    "data_dir",
    "save_dir",
    "model_name",
    "preprocessing_name",
    "batch_size",
    "learning_rate",
    "max_number_of_epochs",
    "max_number_of_steps",
    "save_interval_epochs",
    "save_interval_steps",
    "checkpoint_path",
    "checkpoint_exclude_scopes",
    "sampling_buffer_size",
    "shuffle_buffer_size",
    "train_image_size",
    "shutdown_after_train",
    "eval_batch_size",
    "keep_checkpoint_max",
    "embedding_size",
    "triplet_strategy",
    "margin",
    "l2norm",
    "use_attr",
    "use_attr_net",
    "num_hidden_attr_net",
    "attr_dim",
]
server_map = {"ip-172-31-12-89": "p3.2xlarge", "ip-172-31-29-214": "p3.8xlarge"}


def main(cf, hyper_param_txt, hostname):
    tf.logging.set_verbosity(tf.logging.INFO)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = F.gpu_no
    print("CUDA Visible device", device_lib.list_local_devices())
    start_time = datetime.now().strftime('%Y%m%d%H%M%S')
    start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if not os.path.isdir(cf.save_dir):
        os.makedirs(cf.save_dir)
    f = open(os.path.join(cf.save_dir, "train_parameters_%s.txt" % start_time), mode="w+")
    f.write(hyper_param_txt)

    # inputs_ph = tf.placeholder(tf.float32, [None, cf.train_image_size, cf.train_image_size, cf.train_image_channel],
    #                            name="inputs")
    # labels_ph = tf.placeholder(tf.int32, [None], name="labels")
    tf.set_random_seed(123)

    files = glob.glob(os.path.join(cf.data_dir, "*_train*tfrecord"))
    files.sort()
    assert len(files) > 0
    num_examples = util.count_records(files)
    global_step = tf.Variable(0, trainable=False)

    image_preprocessing_fn = None
    if cf.preprocessing_name:
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(cf.preprocessing_name, is_training=True)

    def train_pre_process(example_proto):
        features = {"image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
                    "image/class/label": tf.FixedLenFeature((), tf.int64, default_value=0),
                    'image/height': tf.FixedLenFeature((), tf.int64, default_value=0),
                    'image/width': tf.FixedLenFeature((), tf.int64, default_value=0)
                    }
        if cf.use_attr:
            features["image/attr"] = tf.VarLenFeature(dtype=tf.int64)

        parsed_features = tf.parse_single_example(example_proto, features)
        image = tf.image.decode_jpeg(parsed_features["image/encoded"], cf.train_image_channel)

        if image_preprocessing_fn is not None:
            image = image_preprocessing_fn(image, cf.train_image_size, cf.train_image_size)
        else:
            image = tf.cast(image, tf.float32)

            image = tf.expand_dims(image, 0)
            image = tf.image.resize_image_with_pad(image, cf.train_image_size, cf.train_image_size)
            # image = tf.image.resize_bilinear(image, [224, 224], align_corners=False)
            image = tf.squeeze(image, [0])

            image = tf.divide(image, 255.0)
            image = tf.subtract(image, 0.5)
            image = tf.multiply(image, 2.0)

        label = parsed_features["image/class/label"]
        if cf.use_attr:
            return image, label, parsed_features["image/attr"]
        else:
            return image, label

    steps_each_epoch = int(num_examples / cf.batch_size)
    if num_examples % cf.batch_size > 0:
        steps_each_epoch += 1
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(train_pre_process, num_parallel_calls=cf.num_preprocessing_threads)
    dataset = dataset.shuffle(cf.shuffle_buffer_size)
    dataset = dataset.repeat()
    dataset = dataset.batch(cf.sampling_buffer_size)
    dataset = dataset.prefetch(cf.sampling_buffer_size)

    iterator = dataset.make_one_shot_iterator()
    # iterator = dataset.make_initializable_iterator()
    if cf.use_attr:
        images, labels, attrs = iterator.get_next()
    else:
        images, labels = iterator.get_next()

    images_ph = tf.placeholder(tf.float32,
                               [cf.batch_size, cf.train_image_size, cf.train_image_size, cf.train_image_channel],
                               name="inputs")
    labels_ph = tf.placeholder(tf.int32, [cf.batch_size], name="labels")
    if cf.use_attr:
        attrs_ph = tf.placeholder(tf.float32, [cf.batch_size, cf.attr_dim], name="attrs")
        if not cf.use_attr_net:
            cf.embedding_size = cf.attr_dim
    else:
        attrs_ph = None
    # seed_ph = tf.placeholder(tf.int64, (), name="shuffle_seed")

    loss_op, end_points, train_op = model_fn.build_model(images_ph, labels_ph, cf, attrs_ph, True, cf.use_attr_net,
                                                         cf.num_hidden_attr_net, num_examples, global_step,
                                                         use_old_model=cf.use_old_model)
    vars = tf.trainable_variables()
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    # Add summaries for end_points.
    for end_point in end_points:
        x = end_points[end_point]
        summaries.add(tf.summary.histogram('activations/' + end_point, x))
        summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                        tf.nn.zero_fraction(x)))

    # Add summaries for losses.
    for loss in tf.get_collection(tf.GraphKeys.LOSSES):
        summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

    # Add summaries for variables.
    for variable in slim.get_model_variables():
        summaries.add(tf.summary.histogram(variable.op.name, variable))

    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    if cf.quantize_delay >= 0:
        tf.contrib.quantize.create_training_graph(quant_delay=cf.quantize_delay)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter(cf.save_dir, sess.graph)

    epoch = 1
    steps = 1
    latest_epoch = 0
    if cf.checkpoint_path is not None and (os.path.isfile(cf.checkpoint_path) or (
      os.path.isdir(cf.checkpoint_path) and tf.train.latest_checkpoint(cf.checkpoint_path) is not None)):
        latest_checkpoint = tf.train.latest_checkpoint(cf.checkpoint_path)
        exclusions = []
        if cf.checkpoint_exclude_scopes:
            exclusions = [scope.strip()
                          for scope in cf.checkpoint_exclude_scopes.split(',')]
        variables_to_restore = []
        for var in slim.get_model_variables():
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    break
            else:
                variables_to_restore.append(var)

        saver_for_restore = tf.train.Saver(var_list=variables_to_restore, max_to_keep=cf.keep_checkpoint_max)
        if os.path.isdir(cf.checkpoint_path) and tf.train.latest_checkpoint(cf.checkpoint_path) is not None:
            cp = tf.train.latest_checkpoint(cf.checkpoint_path)
        else:
            cp = cf.checkpoint_path
        saver_for_restore.restore(sess, cp)
        if os.path.isdir(cf.checkpoint_path) and tf.train.latest_checkpoint(cf.checkpoint_path) is not None:
            latest_epoch = int(os.path.basename(latest_checkpoint).split("-")[1])
            epoch = latest_epoch + 1
            cf.max_number_of_epochs += latest_epoch
        f.write("%s:%s\n" % ("restore_checkpoint", latest_checkpoint))
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=cf.keep_checkpoint_max)
    f.close()
    num_trained_images = 0
    last_saved_epoch = None
    last_saved_step = None
    start_avg_loss_steps = 10
    start_total_loss = 0.
    while True:
        # sess.run(iterator.initializer, feed_dict={seed_ph: steps})
        try:
            start = time.time()
            if cf.use_attr:
                tmp_images, tmp_labels, tmp_attrs = sess.run([images, labels, attrs])
                tmp_attrs = np.reshape(tmp_attrs.values, [cf.sampling_buffer_size, cf.attr_dim])
                tmp_attrs = tmp_attrs.astype(np.float64)
            else:
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
            # print(len(pair_indices))
            # continue
            if len(pair_indices) > cf.batch_size:
                pair_indices = pair_indices[:cf.batch_size]
            elif len(pair_indices) < cf.batch_size:
                pair_indices += list(single_index_map.values())[:cf.batch_size - len(pair_indices)]
            # print(pair_indices)
            batch_images = tmp_images[pair_indices]
            batch_labels = tmp_labels[pair_indices]
            if cf.use_attr:
                batch_attrs = tmp_attrs[pair_indices]

            sampling_time = time.time() - start
            tmp_images = None
            tmp_labels = None
            start = time.time()
            feed_dict = {images_ph: batch_images, labels_ph: batch_labels}
            if cf.use_attr:
                feed_dict[attrs_ph] = batch_attrs
            if steps % cf.save_summaries_steps == 0:
                loss, _, summary = sess.run([loss_op, train_op, summary_op], feed_dict=feed_dict)
                summary_writer.add_summary(summary, steps)
            else:
                loss, _ = sess.run([loss_op, train_op], feed_dict=feed_dict)
            if steps <= start_avg_loss_steps:
                start_total_loss += loss
            train_time = time.time() - start

            if steps % cf.log_every_n_steps == 0:
                now = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                print("[%s: %d epoch(%d/%d), %d steps] sampling time: %f, train time: %f, loss: %f" % (
                    now, epoch, steps % steps_each_epoch, steps_each_epoch, steps, sampling_time, train_time, loss))
            num_trained_images += cf.batch_size

            if cf.use_save_steps:
                if steps % cf.save_interval_steps == 0:
                    saver.save(sess, cf.save_dir + "/model.ckpt", steps)
                    last_saved_step = steps

            if cf.max_number_of_steps is not None and steps >= cf.max_number_of_steps:
                break
            steps += 1

            if num_trained_images >= num_examples:
                if not cf.use_save_steps and cf.save_interval_epochs >= 1 and (
                  epoch - latest_epoch) % cf.save_interval_epochs == 0:
                    saver.save(sess, cf.save_dir + "/model.ckpt", epoch)
                    last_saved_epoch = epoch
                if epoch >= cf.max_number_of_epochs:
                    break
                epoch += 1
                num_trained_images = 0

        except tf.errors.OutOfRangeError:
            break

    if cf.use_save_steps:
        if last_saved_step is None or last_saved_step < steps:
            saver.save(sess, cf.save_dir + "/model.ckpt", steps)
    else:
        if last_saved_epoch is None or last_saved_epoch < epoch:
            saver.save(sess, cf.save_dir + "/model.ckpt", epoch)

    summary_writer.add_summary(sess.run(summary_op, feed_dict=feed_dict), steps)

    sess.close()
    tf.reset_default_graph()

    if cf.notify_after_training:
        txt = "%s[%s]\n\n" % (hostname, socket.gethostbyname(socket.gethostname()))
        txt += "start avg loss : %f" % (start_total_loss / start_avg_loss_steps)
        txt += "last loss : %f" % loss
        txt += "start time: %s\n" % start_time_str
        txt += "end time: %s\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if cf.eval_after_training:
            txt += "going to evaluate"
        else:
            txt += "not going to evaluate"
        txt += "\n[params]\n"
        txt += hyper_param_txt
        util.send_msg_to_slack("\n\n==================================\nTraining is Done\n" + txt)

    if cf.eval_after_training:
        cuda.select_device(0)
        cuda.close()
        eval_cmd = 'python -u multiple_search_models.py --model_dir="%s" --embedding_size=%d --data_dir="%s" --model_name=%s --max_top_k=%d --shutdown_after_train=%d --gpu_no=%s --step_type=%s --image_size=%s --eval_batch_size=%d --preprocessing_name=%s --notify_after_training=%d --use_old_model=%d --save_static_data=%d' % (
            cf.save_dir, cf.embedding_size, cf.data_dir, cf.model_name, cf.eval_max_top_k,
            1 if cf.shutdown_after_train else 0, cf.gpu_no, "step" if cf.use_save_steps else "epoch",
            cf.train_image_size, cf.eval_batch_size, cf.preprocessing_name, 1 if cf.notify_after_training else 0,
            1 if cf.use_old_model else 0, 1 if cf.save_static_data else 0)
        print(eval_cmd)
        os.system(eval_cmd)
    else:
        if cf.shutdown_after_train:
            os.system("sudo shutdown now")


if __name__ == '__main__':
    fl = tf.app.flags

    fl.DEFINE_string('save_dir', 'experiments/test', '')
    fl.DEFINE_integer('num_preprocessing_threads', 4, '')
    fl.DEFINE_integer('log_every_n_steps', 10, 'The frequency with which logs are print.')
    fl.DEFINE_integer('save_summaries_steps', 10, '')
    fl.DEFINE_boolean('use_save_steps', False, '')
    fl.DEFINE_integer('save_interval_steps', 10000, '')
    fl.DEFINE_integer('save_interval_epochs', 2, '')
    fl.DEFINE_boolean('shutdown_after_train', False, '')
    fl.DEFINE_boolean('eval_after_training', True, '')
    fl.DEFINE_integer('eval_max_top_k', 20, '')
    fl.DEFINE_integer('eval_batch_size', 128, '')
    fl.DEFINE_boolean('notify_after_training', True, '')
    fl.DEFINE_boolean('save_static_data', True, '')

    fl.DEFINE_string('gpu_no', "0", '')

    #######################
    # Dataset Flags #
    #######################

    fl.DEFINE_string('data_dir',
                     'D:\data\\fashion\image_retrieval\images_for_tfrecord\street2shop\\tfrecord',
                     '')
    # fl.DEFINE_string('data_dir',
    #                  "D:\data\\fashion\image_retrieval\cafe24product\\tfrecord_with_attr",
    #                  '')

    fl.DEFINE_string('model_name', 'inception_resnet_v2', '')
    fl.DEFINE_string('preprocessing_name', "inception", '')
    fl.DEFINE_integer('batch_size', 16, '')
    fl.DEFINE_integer('sampling_buffer_size', 180, '')
    fl.DEFINE_integer('shuffle_buffer_size', 360, '')
    fl.DEFINE_integer('train_image_channel', 3, '')
    fl.DEFINE_integer('train_image_size', 299, '')
    fl.DEFINE_integer('max_number_of_steps', None, '')
    fl.DEFINE_integer('max_number_of_epochs', 10, '')
    fl.DEFINE_integer('keep_checkpoint_max', 5, '')

    #######################
    # Triplet #
    #######################
    fl.DEFINE_integer('embedding_size', 128, '')
    fl.DEFINE_string('triplet_strategy', 'batch_hard', '')
    fl.DEFINE_float('margin', 0.5, '')
    fl.DEFINE_boolean('squared', False, '')
    fl.DEFINE_boolean('l2norm', False, '')

    #######################
    # Attribute data #
    #######################
    fl.DEFINE_boolean('use_attr', False, '')
    fl.DEFINE_boolean('use_attr_net', False, '')
    fl.DEFINE_integer('num_hidden_attr_net', 1, '')
    fl.DEFINE_integer('attr_dim', 463, '')
    fl.DEFINE_float('attr_loss_weight', 1.0, '')
    fl.DEFINE_boolean('use_old_model', False, '')

    ######################
    # Optimization Flags #
    ######################

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
    fl.DEFINE_integer('quantize_delay', -1, 'Number of steps to start quantized training. Set to -1 would disable')

    #######################
    # Learning Rate Flags #
    #######################
    fl.DEFINE_string('learning_rate_decay_type', 'exponential', '"fixed", "exponential",'' or "polynomial"')
    fl.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    fl.DEFINE_float('end_learning_rate', 0.0001, 'The minimal end learning rate used by a polynomial decay.')
    fl.DEFINE_float('learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
    fl.DEFINE_float('num_epochs_per_decay', 2.0, 'Number of epochs after which learning rate decays.')
    fl.DEFINE_float('moving_average_decay', None, 'The decay to use for the moving average.')

    #####################
    # Fine-Tuning Flags #
    #####################
    # fl.DEFINE_string('checkpoint_path', "D:/pretrained/inception_resnet_v2_2016_08_30.ckpt", '')
    fl.DEFINE_string('checkpoint_path', None, '')
    fl.DEFINE_string('checkpoint_exclude_scopes', None,
                     'Comma-separated list of scopes of variables to exclude when restoring '
                     'from a checkpoint.')
    # fl.DEFINE_string('checkpoint_exclude_scopes', "InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits",
    #                  'Comma-separated list of scopes of variables to exclude when restoring '
    #                  'from a checkpoint.')

    fl.DEFINE_string('trainable_scopes', None, 'Comma-separated list of scopes to filter the set of variables to train.'
                                               'By default, None would train all the variables.')

    F = fl.FLAGS
    param_iterator = iter(F)
    hyper_param_txt = ""
    for key in notify_params:
        hyper_param_txt += "%s:%s\n" % (key, str(getattr(F, key)))
    hostname = socket.gethostname()
    if hostname in server_map:
        hostname = server_map[hostname] + "_" + hostname
    try:
        start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        txt = "%s[%s]\n\n" % (hostname, socket.gethostbyname(socket.gethostname()))
        txt += "start time: %s\n" % start_time
        txt += "\n[params]\n"
        txt += hyper_param_txt
        util.send_msg_to_slack("\n\n==================================\nStarted to train !!!\n\n" + txt)
        main(F, hyper_param_txt, hostname)

        if F.eval_after_training:
            cuda.select_device(0)
            cuda.close()
            eval_cmd = 'python -u multiple_search_models.py --model_dir="%s" --embedding_size=%d --data_dir="%s" --model_name=%s --max_top_k=%d --shutdown_after_train=%d --gpu_no=%s --step_type=%s --image_size=%s --eval_batch_size=%d --preprocessing_name=%s --notify_after_training=%d' % (
                F.save_dir, F.embedding_size, F.data_dir, F.model_name, F.eval_max_top_k,
                1 if F.shutdown_after_train else 0, F.gpu_no, "step" if F.use_save_steps else "epoch",
                F.train_image_size, F.eval_batch_size, F.preprocessing_name, 1 if F.notify_after_training else 0)
            print(eval_cmd)
            os.system(eval_cmd)

    except:
        txt = "%s[%s]\n\n" % (hostname, socket.gethostbyname(socket.gethostname()))
        txt += "start time: %s\n" % start_time
        txt += "end time: %s\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        txt += "\n[stack trace]\n"
        txt += traceback.format_exc()
        txt += "\n[params]\n"
        txt += hyper_param_txt
        util.send_msg_to_slack("\n\n==================================\nTraining Exception!!!\n\n" + txt)
        traceback.print_exc()

    if not F.eval_after_training and F.shutdown_after_train:
        os.system("sudo shutdown now")
