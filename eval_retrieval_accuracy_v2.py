import argparse
import os
import pathlib
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from model import mnist_dataset
from model import tfrecords_dataset
from model.utils import Params
from model.model_fn import model_fn
from model import input_fn
from model import tfrecord_input_fn
import util
from model import model_fn
import glob
from tensorflow.python.client import device_lib

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/alexnet',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='D:\data\deep_fashion\In-shop Clothes Retrieval Benchmark\\tfrecord',
                    help="Directory containing the dataset")
parser.add_argument('--restore_epoch', default=None,
                    help="Directory containing the dataset")
parser.add_argument('--embedding_size', default=128,
                    help="Directory containing the dataset")
parser.add_argument('--image_size', default=128,
                    help="Directory containing the dataset")
parser.add_argument('--model_name', default="tfrecord",
                    help="Directory containing the dataset")
parser.add_argument('--use_attr', default="0",
                    help="Directory containing the dataset")
parser.add_argument('--gpu_no', default="0",
                    help="Directory containing the dataset")
parser.add_argument('--eval_batch_size', default=256,
                    help="Directory containing the dataset")
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    args.use_attr = bool(int(args.use_attr))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no
    print("CUDA Visible device", device_lib.list_local_devices())


    def train_pre_process(example_proto):
        features = {"image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
                    "image/class/label": tf.FixedLenFeature((), tf.int64, default_value=0),
                    'image/height': tf.FixedLenFeature((), tf.int64, default_value=0),
                    'image/width': tf.FixedLenFeature((), tf.int64, default_value=0)
                    }
        if args.use_attr:
            features["image/attr"] = tf.VarLenFeature(dtype=tf.int64)

        parsed_features = tf.parse_single_example(example_proto, features)
        image = tf.image.decode_jpeg(parsed_features["image/encoded"], 3)
        image = tf.cast(image, tf.float32)

        image = tf.expand_dims(image, 0)
        image = tf.image.resize_image_with_pad(image, int(args.image_size), int(args.image_size))
        # image = tf.image.resize_bilinear(image, [224, 224], align_corners=False)
        image = tf.squeeze(image, [0])

        image = tf.divide(image, 255.0)
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)

        label = parsed_features["image/class/label"]
        if args.use_attr:
            return image, label, parsed_features["image/attr"]
        else:
            return image, label


    files_op = tf.placeholder(tf.string, shape=[None], name="files")
    num_examples_op = tf.placeholder(tf.int64, shape=(), name="num_examples")
    dataset = tf.data.TFRecordDataset(files_op)
    dataset = dataset.map(train_pre_process)
    dataset = dataset.batch(num_examples_op)
    iterator = dataset.make_initializable_iterator()
    if args.use_attr:
        images, labels, attrs = iterator.get_next()
    else:
        images, labels = iterator.get_next()
        attrs = None

    embedding_op = model_fn.build_model(images, None, args, attrs, False)

    query_files = glob.glob(os.path.join(args.data_dir, "*_query*tfrecord"))
    query_files.sort()
    assert len(query_files) > 0
    query_num_examples = util.count_records(query_files)
    index_files = glob.glob(os.path.join(args.data_dir, "*_index*tfrecord"))
    index_files.sort()
    assert len(index_files) > 0
    index_num_examples = util.count_records(index_files)

    embedding_batch_size = int(args.eval_batch_size)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())

    if args.restore_epoch is not None:
        restore_epoch = int(args.restore_epoch)
        latste_checkpoint = tf.train.latest_checkpoint(args.model_dir)
        restore_path = os.path.join(os.path.dirname(latste_checkpoint), "model.ckpt-%d" % restore_epoch)
        saver.restore(sess, restore_path)
    else:
        saver.restore(sess, tf.train.latest_checkpoint(args.model_dir))

    sess.run(iterator.initializer, feed_dict={files_op: query_files, num_examples_op: query_num_examples})
    if attrs is None:
        query_labels = sess.run(labels)
    else:
        query_labels, query_attrs = sess.run([labels, attrs])

        print("query attrs", query_attrs.shape)
    print("query labels", query_labels.shape)
    query_embeddings = np.zeros((query_num_examples, int(args.embedding_size)))
    steps = query_num_examples / embedding_batch_size
    if query_num_examples % embedding_batch_size > 0:
        steps += 1
    sess.run(iterator.initializer, feed_dict={files_op: query_files, num_examples_op: embedding_batch_size})
    for i in range(steps):
        tmp_query_embeddings = sess.run(embedding_op)
        print("query embeddings", tmp_query_embeddings.shape)
        for j, tmp_qe in enumerate(tmp_query_embeddings):
            query_embeddings[i * embedding_batch_size + j] = tmp_qe
    print(query_embeddings.shape)

    sess.run(iterator.initializer, feed_dict={files_op: index_files, num_examples_op: index_num_examples})
    if attrs is None:
        index_labels = sess.run(labels)
    else:
        index_labels, index_attrs = sess.run([labels, attrs])

        print("index attrs", index_attrs.shape)
    print("index labels", index_labels.shape)
    index_embeddings = np.zeros((index_num_examples, int(args.embedding_size)))
    steps = index_num_examples / embedding_batch_size
    if index_num_examples % embedding_batch_size > 0:
        steps += 1
    sess.run(iterator.initializer, feed_dict={files_op: index_files, num_examples_op: embedding_batch_size})
    for i in range(steps):
        tmp_index_embeddings = sess.run(embedding_op)
        print("index embeddings", tmp_index_embeddings.shape)
        for j, tmp_ie in enumerate(tmp_index_embeddings):
            index_embeddings[i * embedding_batch_size + j] = tmp_ie
    print(index_embeddings.shape)

    sess.close()
    tf.reset_default_graph()

    np.save(os.path.join(args.model_dir, "query_embeddings.npy"), query_embeddings)
    np.save(os.path.join(args.model_dir, "index_embeddings.npy"), index_embeddings)
    np.save(os.path.join(args.model_dir, "query_labels.npy"), query_labels)
    np.save(os.path.join(args.model_dir, "index_labels.npy"), index_labels)
    if attrs is not None:
        np.save(os.path.join(args.model_dir, "query_attrs.npy"), query_attrs)
        np.save(os.path.join(args.model_dir, "index_attrs.npy"), index_attrs)
