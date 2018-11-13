"""Train the model"""

import argparse
import os

import tensorflow as tf

from model import input_fn
from model import tfrecord_input_fn
from model.model_fn import model_fn
from model.utils import Params
import glob
import util

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/alexnet',
                    help="Experiment directory containing params.json")
# parser.add_argument('--model_dir', default='experiments/alexnet',
#                     help="Experiment directory containing params.json")
# parser.add_argument('--data_dir', default='D:/data/deep_fashion/In-shop Clothes Retrieval Benchmark/tfrecord',
#                     help="Directory containing the dataset")
# parser.add_argument('--data_dir', default='D:\data\pattern\img',
#                     help="Directory containing the dataset")
parser.add_argument('--data_dir', default='D:/data/fashion/image_retrieval/cafe24product')
parser.add_argument('--dataset_name', default='tfrecord',
                    help="Directory containing the dataset")
# parser.add_argument('--dataset_name', default='mnist',
#                     help="Directory containing the dataset")
parser.add_argument('--save_checkpoints_steps', default=None,
                    help="Directory containing the dataset")
parser.add_argument('--keep_checkpoint_max', default=20,
                    help="Directory containing the dataset")
parser.add_argument('--num_class_sampling', default=32,
                    help="Directory containing the dataset")
parser.add_argument('--num_image_sampling', default=4,
                    help="Directory containing the dataset")
parser.add_argument('--save_checkpoints_epochs', default=1,
                    help="Directory containing the dataset")

if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)
    random_seed = 230
    tf.set_random_seed(random_seed)
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    params.batch_size = int(args.num_class_sampling) * int(args.num_image_sampling)

    assert args.save_checkpoints_steps is not None or args.save_checkpoints_epochs is not None

    if args.save_checkpoints_epochs is not None:
        files = glob.glob(os.path.join(args.data_dir, "*_train_*tfrecord"))
        num_examples = util.count_records(files)
        steps_each_epoch = num_examples // params.batch_size
        save_checkpoints_steps = steps_each_epoch * int(args.save_checkpoints_epochs)
    else:
        save_checkpoints_steps = int(args.save_checkpoints_steps)

    # Define the model
    tf.logging.info("Creating the model...")
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    config = tf.estimator.RunConfig(tf_random_seed=random_seed,
                                    model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps,
                                    save_checkpoints_steps=save_checkpoints_steps,
                                    session_config=tf_config,
                                    keep_checkpoint_max=int(args.keep_checkpoint_max))

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
    # tf.logging.info("Evaluation on test set.")
    # res = estimator.evaluate(lambda: test_input_fn(args.data_dir, params))
    # for key in res:
    #     print("{}: {}".format(key, res[key]))
