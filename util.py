import tensorflow as tf
import numpy as np
import sys
import socket

if sys.version_info[0] < 3:
    import urllib2 as request
else:
    from urllib import request
import json

F = tf.app.flags.FLAGS


def set_flags(conf):
    backup = {}
    for key in conf:
        val = conf[key]
        if hasattr(F, key):
            backup[key] = val
            setattr(F, key, val)
    return backup


def restore_flags(bak_conf):
    for key in bak_conf:
        val = bak_conf[key]
        if hasattr(F, key):
            setattr(F, key, val)


def get_attr(file_path, func_name):
    try:
        module = __import__(file_path + "", globals(), locals(), [func_name])
        return getattr(module, func_name)

    except ImportError:
        return None
    except AttributeError:
        return None


def count_records(tfrecord_filenames):
    c = 0
    for fn in tfrecord_filenames:
        for _ in tf.python_io.tf_record_iterator(fn):
            c += 1
    return c


def send_msg_to_slack(text):
    post = {"text": text}

    try:
        json_data = json.dumps(post)
        req = request.Request("https://hooks.slack.com/services/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",
                              data=json_data.encode(),
                              headers={'Content-Type': 'application/json'})
        resp = request.urlopen(req)
    except Exception as em:
        print("EXCEPTION: " + str(em))


def get_images_by_indices(tfrecord_filenames, indices):
    indices = {i: True for i in indices}
    tfrecord_filenames.sort()
    image_list = []
    total = 0
    for fn in tfrecord_filenames:
        record_iterator = tf.python_io.tf_record_iterator(path=fn)

        for i, string_record in enumerate(record_iterator):
            j = i + total
            if j not in indices:
                continue
            example = tf.train.Example()
            example.ParseFromString(string_record)
            image_string = example.features.feature['image/encoded'].bytes_list.value[0].decode('utf-8')
            height = int(example.features.feature['image/height'].int64_list.value[0])
            width = int(example.features.feature['image/width'].int64_list.value[0])
            img_1d = np.fromstring(image_string, dtype=np.uint8)
            reconstructed_img = img_1d.reshape((height, width, -1))
            image_list.append(reconstructed_img)
        total += (i + 1)
    return image_list


def count_records_each_class(tfrecords_filenames):
    labels = {}
    for tfrecords_filename in tfrecords_filenames:
        record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            label = int(example.features.feature['image/class/label'].int64_list.value[0])
            if not label in labels:
                labels[label] = 0
            labels[label] += 1
    return labels


def create_label_map(tfrecords_filenames):
    labels = {}
    for tfrecords_filename in tfrecords_filenames:
        record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            label = example.features.feature['image/class/name'].bytes_list.value[0].decode('utf-8')
            if label not in labels:
                labels[label] = len(labels)
    return labels


def count_label_cnt(tfrecords_filenames):
    labels = {}
    for tfrecords_filename in tfrecords_filenames:
        record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            label = example.features.feature['image/class/name'].bytes_list.value[0].decode('utf-8')
            if label not in labels:
                labels[label] = True
    return len(labels)
