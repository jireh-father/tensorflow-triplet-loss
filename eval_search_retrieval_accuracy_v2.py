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

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/alexnet',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='D:\data\deep_fashion\In-shop Clothes Retrieval Benchmark\\tfrecord',
                    help="Directory containing the dataset")
parser.add_argument('--save', default='0',
                    help="Directory containing the dataset")
parser.add_argument('--max_top_k', default=100,
                    help="Directory containing the dataset")
parser.add_argument('--embedding_size', default=128,
                    help="Directory containing the dataset")
parser.add_argument('--model_name', default="tfrecord",
                    help="Directory containing the dataset")
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)


    def train_pre_process(example_proto):
        features = {"image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
                    "image/class/label": tf.FixedLenFeature((), tf.int64, default_value=0)}
        parsed_features = tf.parse_single_example(example_proto, features)
        image = tf.image.decode_jpeg(parsed_features["image/encoded"], 3)
        image = tf.cast(image, tf.float32)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_image_with_pad(image, 224, 224)
        image = tf.squeeze(image, [0])
        image = tf.divide(image, 255.0)
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image, parsed_features["image/class/label"]


    files_op = tf.placeholder(tf.string, shape=[None], name="files")
    num_examples_op = tf.placeholder(tf.int64, shape=(), name="num_examples")
    dataset = tf.data.TFRecordDataset(files_op)
    dataset = dataset.map(train_pre_process)
    dataset = dataset.batch(num_examples_op)
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()

    embedding_op = model_fn.build_model(images, None, args, False)

    query_files = glob.glob(os.path.join(args.data_dir, "*_query*tfrecord"))
    query_files.sort()
    assert len(query_files) > 0
    query_num_examples = util.count_records(query_files)
    index_files = glob.glob(os.path.join(args.data_dir, "*_index*tfrecord"))
    index_files.sort()
    assert len(index_files) > 0
    index_num_examples = util.count_records(index_files)

    embedding_batch_size = 512

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, tf.train.latest_checkpoint(args.model_dir))

    sess.run(iterator.initializer, feed_dict={files_op: query_files, num_examples_op: query_num_examples})
    query_labels = sess.run(labels)
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
    index_labels = sess.run(labels)
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

    if args.save != "0":
        np.save(os.path.join(args.model_dir, "query_embeddings.npy"), query_embeddings)
        np.save(os.path.join(args.model_dir, "index_embeddings.npy"), index_embeddings)
        np.save(os.path.join(args.model_dir, "query_labels.npy"), query_labels)
        np.save(os.path.join(args.model_dir, "index_labels.npy"), index_labels)

    import faiss

    ngpus = faiss.get_num_gpus()

    print("number of GPUs:", ngpus)
    cpu_index = faiss.IndexFlatL2(int(args.embedding_size))

    gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
        cpu_index
    )
    max_top_k = args.max_top_k
    gpu_index.add(index_embeddings)  # add vectors to the index
    print(gpu_index.ntotal)

    k = max_top_k  # we want to see 4 nearest neighbors
    print("start search!")
    search_d, search_idx = gpu_index.search(query_embeddings, k)
    print("end search!")
    accuracy_list = []
    accuracies = []
    for i in range(max_top_k):
        accuracy_list.append([0, 0, 0])
    true_indices = []
    false_indices = []

    for i, q_label in enumerate(query_labels):
        searched_indices = index_labels[search_idx[i]]
        searched_distances = search_d[searched_indices]
        tmp_dist = [str(d) for d in search_d[i]]
        is_true = False
        for j in range(1, max_top_k + 1):
            tmp_indices = searched_indices[:j]
            if q_label in tmp_indices:
                accuracy_list[j - 1][0] += 1
                if not is_true:
                    true_indices.append([i, j, list(search_idx[i]), tmp_dist])
                    is_true = True
            else:
                accuracy_list[j - 1][1] += 1
        if not is_true:
            false_indices.append([i, list(search_idx[i]), tmp_dist])
    for accuracy in accuracy_list:
        accuracy[2] = float(accuracy[0]) / float(len(query_labels))
        accuracies.append(accuracy[2])
    # np.save(os.path.join(args.model_dir, "true_indices.npy"), true_indices)
    import json

    json.dump(true_indices, open(os.path.join(args.model_dir, "true_indices.json"), "w+"))

    json.dump(false_indices, open(os.path.join(args.model_dir, "false_indices.json"), "w+"))
    json.dump(accuracy_list, open(os.path.join(args.model_dir, "accuracy_list.json"), "w+"))
    json.dump(accuracies, open(os.path.join(args.model_dir, "accuracies.json"), "w+"))
    print(accuracy_list)
    print("top %s accuracy" % args.max_top_k,
          float(accuracy_list[int(args.max_top_k) - 1][0]) / float(len(query_labels)))
