import tensorflow as tf

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
