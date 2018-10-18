"""Train the model"""

import argparse
import os

import tensorflow as tf

from model import input_fn
from model import tfrecord_input_fn
from model.model_fn import model_fn
from model.utils import Params

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Experiment directory containing params.json")
# parser.add_argument('--model_dir', default='experiments/alexnet',
#                     help="Experiment directory containing params.json")
# parser.add_argument('--data_dir', default='D:/data/deep_fashion/In-shop Clothes Retrieval Benchmark/tfrecord',
#                     help="Directory containing the dataset")
parser.add_argument('--data_dir', default='./data/mnist',
                    help="Directory containing the dataset")
# parser.add_argument('--dataset_name', default='tfrecord',
#                     help="Directory containing the dataset")
parser.add_argument('--dataset_name', default='mnist',
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
    if args.dataset_name == "tfrecord":
        train_input_fn = tfrecord_input_fn.train_input_fn
        test_input_fn = tfrecord_input_fn.test_input_fn
    else:
        train_input_fn = input_fn.train_input_fn
        test_input_fn = input_fn.test_input_fn

    # Train the model
    tf.logging.info("Starting training for {} epoch(s).".format(params.num_epochs))
    estimator.train(lambda: train_input_fn(args.data_dir, params))

    # Evaluate the model on the test set
    tf.logging.info("Evaluation on test set.")
    res = estimator.evaluate(lambda: test_input_fn(args.data_dir, params))
    for key in res:
        print("{}: {}".format(key, res[key]))
