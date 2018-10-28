import tensorflow as tf
import os
import util


def build_dataset(file_names, preprocessing_name, shuffle_buffer_size, sampling_buffer_size, num_map_parallel):
    ds = tf.data.TFRecordDataset(file_names)

    assert preprocessing_name is not None
    assert os.path.isfile(os.path.join("./preprocessing", preprocessing_name + ".py"))
    preprocessing_f = util.get_attr('preprocessing.%s' % preprocessing_name, "train_preprocessing")

    ds = ds.map(preprocessing_f, num_map_parallel)
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.batch(sampling_buffer_size)
    return ds.prefetch(sampling_buffer_size)
