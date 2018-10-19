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

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/alexnet',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='D:\data\deep_fashion\In-shop Clothes Retrieval Benchmark\\tfrecord',
                    help="Directory containing the dataset")

if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Define the model
    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps)
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    # Compute embeddings on the test set
    tf.logging.info("Predicting")
    index_predictions = estimator.predict(lambda: tfrecord_input_fn.index_input_fn(args.data_dir, params))
    query_predictions = estimator.predict(lambda: tfrecord_input_fn.query_input_fn(args.data_dir, params))

    index_label_ds = tfrecord_input_fn.index_label_fn(args.data_dir, params)
    index_iterator = index_label_ds.make_one_shot_iterator()
    index_labels = index_iterator.get_next()
    query_label_ds = tfrecord_input_fn.query_label_fn(args.data_dir, params)
    query_iterator = query_label_ds.make_one_shot_iterator()
    query_labels = query_iterator.get_next()

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session()
    all_index_labels, all_query_labels = sess.run([index_labels, query_labels])
    sess.close()

    query_embeddings = np.zeros((params.query_cnt, params.embedding_size))
    total = 0
    for i, p in enumerate(query_predictions):
        total += 1
        query_embeddings[i] = p['embeddings']
    print("query total", total)
    index_embeddings = np.zeros((params.index_cnt, params.embedding_size))
    total = 0
    for i, p in enumerate(index_predictions):
        total += 1
        index_embeddings[i] = p['embeddings']
    print("index total", total)
    print(query_embeddings.shape, index_embeddings.shape)

    import faiss

    ngpus = faiss.get_num_gpus()

    print("number of GPUs:", ngpus)
    cpu_index = faiss.IndexFlatL2(params.embedding_size)

    gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
        cpu_index
    )

    gpu_index.add(index_embeddings)  # add vectors to the index
    print(gpu_index.ntotal)

    k = 20  # we want to see 4 nearest neighbors
    print("start search!")
    search_d, search_idx = gpu_index.search(query_embeddings, k)
    print("end search!")
    true_cnt = 0
    false_cnt = 0
    for i in range(params.query_cnt):
        searched_indices = index_labels[search_idx[i]]
        print(searched_indices)
        if query_labels[i] in searched_indices:
            true_cnt += 1
        else:
            false_cnt += 1
    print("true", true_cnt, "false", false_cnt)
