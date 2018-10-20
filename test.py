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
args = parser.parse_args()
json_path = os.path.join(args.model_dir, 'params.json')
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
params = Params(json_path)
data_dir = "D:\data\deep_fashion\In-shop Clothes Retrieval Benchmark\\tfrecord"
index_label_ds = tfrecord_input_fn.index_input_fn(data_dir, params)
index_iterator = index_label_ds.make_one_shot_iterator()
index_labels = index_iterator.get_next()

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session()
images = sess.run(index_labels)
sess.close()
print(images[0].shape)
print(images[0].max())
print(images[0].min())
from PIL import Image
im = Image.fromarray(images[0][0].astype('uint8'))
im.show()