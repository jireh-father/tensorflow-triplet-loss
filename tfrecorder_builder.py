import tensorflow as tf
import os
import math, json
import sys
from core import dataset_utils
import shutil
import glob
from multiprocessing import Process
import numpy as np


def _get_dataset_filename(dataset_name, output_dir, phase_name, shard_id, num_shards):
    output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (dataset_name, phase_name, shard_id, num_shards)
    return os.path.join(output_dir, output_filename)


def _dataset_exists(dataset_name, output_dir, phase_name, num_shards):
    for shard_id in range(num_shards):
        output_filename = _get_dataset_filename(
            dataset_name, output_dir, phase_name, shard_id, num_shards)
        if not tf.gfile.Exists(output_filename):
            return False
    return True


def _get_filenames_and_classes(image_dir):
    root = image_dir
    directories = []
    class_names = []
    for filename in os.listdir(root):
        path = os.path.join(root, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)

    photo_filenames = []
    # todo: handle both uppercase and lowercase
    if os.name == 'nt':
        exts = ["jpg", "jpeg"]
    else:
        exts = ["jpg", "JPEG", "JPG", "jpeg"]
    for directory in directories:
        for ext in exts:
            for path in glob.glob(os.path.join(directory, "*.%s" % ext)):
                # path = os.path.join(directory, filename)
                photo_filenames.append(path)

    return sorted(photo_filenames), sorted(class_names)


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self, num_channels):
        # Initializes function that decodes Grayscale JPEG data.
        self.num_channels = num_channels
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=num_channels)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == self.num_channels
        return image


def _convert_dataset(dataset_name, phase_name, filenames, class_names_to_ids, output_dir, num_shards,
                     num_channels=3, attr_map=None):
    """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
    num_per_shard = int(math.ceil(len(filenames) / float(num_shards)))

    with tf.Graph().as_default():
        image_reader = ImageReader(num_channels)

        with tf.Session('') as sess:

            for shard_id in range(num_shards):
                assert not os.path.isfile(output_dir)
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)
                output_filename = _get_dataset_filename(dataset_name, output_dir, phase_name, shard_id, num_shards)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d, %s' % (
                            i + 1, len(filenames), shard_id, filenames[i]))
                        sys.stdout.flush()

                        # Read the filename:
                        image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
                        try:
                            height, width = image_reader.read_image_dims(sess, image_data)
                        except:
                            continue

                        class_name = os.path.basename(os.path.dirname(filenames[i]))
                        class_id = class_names_to_ids[class_name]
                        attr = None
                        if attr_map is not None:
                            attr = attr_map[os.path.basename(filenames[i])]

                        ext = 'jpg'
                        if sys.version_info[0] == 3:
                            ext = ext.encode()
                            class_name = class_name.encode()

                        example = dataset_utils.image_to_tfexample(image_data, ext, height, width, class_id, class_name,
                                                                   attr)
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def _clean_up_temporary_files(dataset_dir):
    """Removes temporary files used to create the dataset.

    Args:
      dataset_dir: The directory where the temporary files are stored.
    """

    dirs = os.listdir(dataset_dir)
    for tmp_dir in dirs:
        shutil.rmtree(tmp_dir)


def make_tfrecords(dataset_name, phase_name, image_dir, output_dir, num_shards, num_channels, remove_images,
                   attr_path=None):
    if not tf.gfile.Exists(image_dir):
        tf.gfile.MakeDirs(image_dir)

    if _dataset_exists(dataset_name, output_dir, phase_name, num_shards):
        print('Dataset files already exist. Exiting without re-creating them.')
        return False

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    photo_filenames, class_names = _get_filenames_and_classes(image_dir)

    np.save(os.path.join(output_dir, "file_names_%s.npy" % phase_name), photo_filenames)

    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    # todo: add bounding box, landmarks, etc data
    # First, convert the training and validation sets.
    attr_map = None
    if attr_path is not None:
        assert os.path.isfile(attr_path)
        attr_map = json.load(open(attr_path))
    _convert_dataset(dataset_name, phase_name, photo_filenames, class_names_to_ids, output_dir, num_shards,
                     num_channels, attr_map)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    json.dump(labels_to_class_names, open(os.path.join(output_dir, "labels_%s.json" % phase_name), "w+"))
    dataset_utils.write_label_file(labels_to_class_names, image_dir, dataset_name)

    if remove_images:
        _clean_up_temporary_files(image_dir)
    return True


if __name__ == '__main__':
    fl = tf.app.flags
    fl.DEFINE_boolean('parallel_exec', False, '')
    fl.DEFINE_boolean('multiple', True, '')

    fl.DEFINE_string('dataset_name', "deepfashion", "")
    fl.DEFINE_string('phase_name', "train", "")
    fl.DEFINE_string('image_dir',
                     'D:/data/fashion/image_retrieval/deep_fashion/In-shop Clothes Retrieval Benchmark/split', '')
    fl.DEFINE_string('tfrecord_output',
                     'D:/data/fashion/image_retrieval/deep_fashion/In-shop Clothes Retrieval Benchmark/tfrecord_with_attr',
                     '')
    fl.DEFINE_string('attr_path',
                     'D:/data/fashion/image_retrieval/deep_fashion/In-shop Clothes Retrieval Benchmark/Anno/attr_file_map.json',
                     '')
    fl.DEFINE_integer('num_channels', 3, '')
    fl.DEFINE_integer('num_shards', 4, '')
    fl.DEFINE_boolean('remove_images', False, '')

    F = tf.app.flags.FLAGS

    if F.multiple:
        image_dirs = glob.glob(os.path.join(F.image_dir, "*"))
        for image_dir in image_dirs:
            p = Process(target=make_tfrecords, args=(
                F.dataset_name, os.path.basename(image_dir), image_dir, F.tfrecord_output, F.num_shards,
                F.num_channels, F.remove_images, F.attr_path))
            print("started to build tfrecords: %s" % (image_dir))
            p.start()
            if not F.parallel_exec:
                p.join()
    else:
        make_tfrecords(F.dataset_name, F.phase_name, F.image_dir, F.tfrecord_output, F.num_shards, F.num_channels,
                       F.remove_images, F.attr_path)
