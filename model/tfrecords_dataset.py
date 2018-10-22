#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import os, glob
import tensorflow as tf


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

    return image, label


def test_pre_process(example_proto):
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

    return image, label


def only_label(example_proto):
    features = {
        "image/class/label": tf.FixedLenFeature((), tf.int64, default_value=0),
    }

    parsed_features = tf.parse_single_example(example_proto, features)

    label = parsed_features["image/class/label"]

    return label


def dataset(tfrecord_files, preprocess_fn):
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    return dataset.map(preprocess_fn)


def train(directory):
    files = glob.glob(os.path.join(directory, "*_train_*tfrecord"))
    files.sort()
    assert len(files) > 0
    return dataset(files, train_pre_process)


def test(directory):
    files = glob.glob(os.path.join(directory, "*_test_*tfrecord"))
    files.sort()
    assert len(files) > 0
    return dataset(files, test_pre_process)


def query(directory):
    files = glob.glob(os.path.join(directory, "*_query_*tfrecord"))
    files.sort()
    assert len(files) > 0
    return dataset(files, test_pre_process)


def index(directory):
    files = glob.glob(os.path.join(directory, "*_index_*tfrecord"))
    files.sort()
    assert len(files) > 0
    return dataset(files, test_pre_process)


def query_label(directory):
    files = glob.glob(os.path.join(directory, "*_query_*tfrecord"))
    files.sort()
    assert len(files) > 0
    return dataset(files, only_label)


def index_label(directory):
    files = glob.glob(os.path.join(directory, "*_index_*tfrecord"))
    files.sort()
    assert len(files) > 0
    return dataset(files, only_label)


def train_label(directory):
    files = glob.glob(os.path.join(directory, "*_test_*tfrecord"))
    files.sort()
    assert len(files) > 0
    return dataset(files, only_label)


def test_label(directory):
    files = glob.glob(os.path.join(directory, "*_test_*tfrecord"))
    files.sort()
    assert len(files) > 0
    return dataset(files, only_label)
