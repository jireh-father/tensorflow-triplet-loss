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

    # Compute embeddings on the train set
    tf.logging.info("Predicting")
    train_predictions = estimator.predict(lambda: tfrecord_input_fn.train_input_fn(args.data_dir, params))

    train_label_ds = tfrecord_input_fn.train_label_fn(args.data_dir, params)
    train_iterator = train_label_ds.make_one_shot_iterator()
    train_labels = train_iterator.get_next()

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session()
    all_train_labels = sess.run(train_labels)
    sess.close()
    print("train_labels", all_train_labels.shape)

    train_embeddings = np.zeros((params.query_cnt, params.embedding_size))
    total = 0
    for i, p in enumerate(train_predictions):
        total += 1
        train_embeddings[i] = p['embeddings']
    print("train total", total)

    np.save(os.path.join(args.model_dir, "train_embeddings.npy"), train_embeddings)
    np.save(os.path.join(args.model_dir, "train_labels.npy"), all_train_labels)
