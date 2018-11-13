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

if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    model_files = glob.glob(os.path.join(args.model_dir, "model.ckpt-*.data*"))
    for model_file in model_files:
        cur_cp_name = "model.%s" % model_file.split(".")[1]
        checkpoint_list = open(os.path.join(args.model_dir, "checkpoint")).readlines()
        checkpoint_list[0] = 'model_checkpoint_path: "%s"' % cur_cp_name
        open(os.path.join(args.model_dir, "checkpoint"), mode="w+").writelines(checkpoint_list)

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

        index_label_ds, index_cnt = tfrecord_input_fn.index_label_fn(args.data_dir, params)
        index_iterator = index_label_ds.make_one_shot_iterator()
        index_labels = index_iterator.get_next()
        query_label_ds, query_cnt = tfrecord_input_fn.query_label_fn(args.data_dir, params)
        query_iterator = query_label_ds.make_one_shot_iterator()
        query_labels = query_iterator.get_next()

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session()
        all_index_labels, all_query_labels = sess.run([index_labels, query_labels])
        sess.close()
        print("index_labels", all_index_labels.shape)
        print("query_labels", all_query_labels.shape)

        query_embeddings = np.zeros((query_cnt, params.embedding_size))
        total = 0
        for i, p in enumerate(query_predictions):
            total += 1
            query_embeddings[i] = p['embeddings']
        print("query total", total)
        index_embeddings = np.zeros((index_cnt, params.embedding_size))
        total = 0
        for i, p in enumerate(index_predictions):
            total += 1
            index_embeddings[i] = p['embeddings']
        print("index total", total)
        print(query_embeddings.shape, index_embeddings.shape)

        if args != "0":
            np.save(os.path.join(args.model_dir, "query_embeddings.npy"), query_embeddings)
            np.save(os.path.join(args.model_dir, "index_embeddings.npy"), index_embeddings)
            np.save(os.path.join(args.model_dir, "query_labels.npy"), all_query_labels)
            np.save(os.path.join(args.model_dir, "index_labels.npy"), all_index_labels)

        tf.reset_default_graph()

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
        gpu_index.reset()
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
