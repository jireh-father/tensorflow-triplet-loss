import tensorflow as tf

F = tf.app.flags.FLAGS
label_map = {}


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
