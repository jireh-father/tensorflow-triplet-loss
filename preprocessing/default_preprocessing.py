import tensorflow as tf
import util


def train_preprocessing(example_proto):
    features = {"image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
                "image/class/label": tf.FixedLenFeature((), tf.int64, default_value=0),
                "image/class/name": tf.FixedLenFeature((), tf.string, default_value=""),
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

    label_name = parsed_features["image/class/name"]

    label = parsed_features["image/class/label"]

    return image, label, label_name


def eval_preprocessing():
    pass
