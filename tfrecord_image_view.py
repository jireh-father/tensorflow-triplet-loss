import argparse
import os
import pathlib
import shutil
import sys
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
import glob


def train_pre_process(example_proto):
    features = {"image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
                "image/class/label": tf.FixedLenFeature((), tf.int64, default_value=0),
                'image/height': tf.FixedLenFeature((), tf.int64, default_value=0),
                'image/width': tf.FixedLenFeature((), tf.int64, default_value=0)
                }
    parsed_features = tf.parse_single_example(example_proto, features)

    image = tf.image.decode_jpeg(parsed_features["image/encoded"], 3)
    image = tf.cast(image, tf.float32)

    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [224, 224], align_corners=False)
    image = tf.squeeze(image, [0])

    image = tf.divide(image, 255.0)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    label = parsed_features["image/class/label"]
    # return parsed_features["image/encoded"], label
    return image, label


def aa(image, label):
    image = tf.image.decode_jpeg(image, 3)
    image = tf.cast(image, tf.float32)

    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [224, 224], align_corners=False)
    image = tf.squeeze(image, [0])

    image = tf.divide(image, 255.0)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image, label


# sampling_p = tf.random_uniform([4], minval=0, maxval=9, dtype=tf.int64)
# sampling_p = tf.placeholder(tf.int64, [4], name="sampling")

data_dir = "c:\source/tensorflow-image-classification-framework/mnist"
files = glob.glob(os.path.join(data_dir, "*_train_*tfrecord"))
aaa = 4
dataset = tf.data.TFRecordDataset(files)
dataset = dataset.map(train_pre_process)
# dataset = dataset.filter(
#     lambda im, lb: tf.reduce_any(tf.equal(sampling_p, lb))
# )
# dataset = dataset.map(aa)
dataset = dataset.shuffle(100)  # whole dataset into the buffer
dataset = dataset.repeat(3)
dataset = dataset.batch(512)
dataset = dataset.prefetch(32)

# index_iterator = dataset.make_initializable_iterator()
index_iterator = dataset.make_one_shot_iterator()
img, index_labels = index_iterator.get_next()
# index_labels = index_iterator.get_next()

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
print(11)
sess = tf.Session(config=tf_config)
print(22)
import time

start = time.time()
# sess.run(index_iterator.initializer, feed_dict={sampling_p: np.array([1, 2, 3, 4], np.int64)})
print(time.time() - start)
ii, ll = sess.run([img, index_labels])
print(ii, ll)
sys.exit()
# from PIL import Image
#
# im = Image.fromarray(ii[0].astype('uint8'))
# im.show()
print(ll)
# ll = sess.run(index_labels)
# print(ll)
# ll = sess.run(index_labels)
# print(ll)
sys.exit()

index_label_ds, shuffle_rand_seed_ph = tfrecord_input_fn.train_input_fn(data_dir, params)
# index_iterator = index_label_ds.make_one_shot_iterator()
index_iterator = index_label_ds.make_initializable_iterator()
img, index_labels = index_iterator.get_next()

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session()
from PIL import Image

for i in range(2):
    sess.run(index_iterator.initializer, feed_dict={shuffle_rand_seed_ph: i})
    j = 0
    while True:
        try:
            images = sess.run(img)
            im = Image.fromarray(images[0].astype('uint8'))
            # im.show()
            # if j == 0:
            im.save("%d.jpg" % i)
            break
            # j += 1
            # if i == 1:
            #     break
        except tf.errors.OutOfRangeError:
            break
sys.exit()
sess.run(index_iterator.initializer)
images = sess.run(img)
sess.close()
print(images[0].shape)
print(images[0].max())
print(images[0].min())

im = Image.fromarray(images[0].astype('uint8'))
im.show()
