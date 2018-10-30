import tensorflow as tf
# from tensorflow.contrib.data.python.ops.interleave_ops import DirectedInterleaveDataset

import model.mnist_dataset as mnist_dataset


# Define the data pipeline
# mnist = mnist_dataset.train(args.data_dir)
#
# datasets = [mnist.filter(lambda img, lab: tf.equal(lab, i)) for i in range(params.num_labels)]
#
#
# def generator():
#     while True:
#         # Sample the labels that will compose the batch
#         labels = np.random.choice(range(params.num_labels),
#                                   params.num_classes_per_batch,
#                                   replace=False)
#         for label in labels:
#             for _ in range(params.num_images_per_class):
#                 yield label
#
#
# selector = tf.data.Dataset.from_generator(generator, tf.int64)
# dataset = DirectedInterleaveDataset(selector, datasets)
#
# batch_size = params.num_classes_per_batch * params.num_images_per_class
# dataset = dataset.batch(batch_size)
# dataset = dataset.prefetch(1)


def _make_balanced_batched_dataset(datasets, num_classes, num_classes_per_batch,
                                   num_images_per_class):
    """Create a dataset with balanced batches sampling from multiple datasets.
    For instance if we have 3 datasets representing classes 0, 1 and 2, and we want to create
    batches containing 2 different classes with 3 images each, the labels of a batch could be:
        2, 2, 2, 0, 0, 0
    Or:
        1, 1, 1, 2, 2, 2
    The total batch size in this case is 6.
    Args:
        datasets: (list of Datasets) the datasets to sample from
        num_classes: (int) number of classes, each dataset represents one class
        num_classes_per_batch: (int) number of different classes composing a batch
        num_images_per_class: (int) number of different images from a class in a batch
    """
    assert len(datasets) == num_classes, \
        "There should be {} datasets, got {}".format(num_classes, len(datasets))

    # def generator():
    #     while True:
    #         # Sample the labels that will compose the batch
    #         labels = np.random.choice(range(num_classes),
    #                                   num_classes_per_batch,
    #                                   replace=False)
    #         for label in labels:
    #             for _ in range(num_images_per_class):
    #                 yield label

    # selector = tf.data.Dataset.from_generator(generator, tf.int64)
    seed_ph = tf.placeholder(tf.int64, [], "seed")

    def generator(_):
        # Sample `num_classes_per_batch` classes for the batch
        sampled = tf.random_shuffle(tf.range(num_classes), 1)[:num_classes_per_batch]
        # Repeat each element `num_images_per_class` times
        batch_labels = tf.tile(tf.expand_dims(sampled, -1), [1, num_images_per_class])
        return tf.to_int64(tf.reshape(batch_labels, [-1]))

    selector = tf.contrib.data.Counter().map(generator)
    selector = selector.apply(tf.contrib.data.unbatch())

    dataset = tf.contrib.data.choose_from_datasets(datasets, selector)

    # dataset = DirectedInterleaveDataset(selector, datasets)

    # Batch
    batch_size = num_classes_per_batch * num_images_per_class
    dataset = dataset.batch(batch_size)

    return dataset, seed_ph


def balanced_train_input_fn(dataset):
    """Train input function for the MNIST dataset with balanced batches.
    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """

    # pylint: disable=cell-var-from-loop
    datasets = [dataset.filter(lambda img, lab: tf.equal(lab, i)) for i in range(3)]

    dataset, seed_ph = _make_balanced_batched_dataset(datasets,
                                                      3,
                                                      3,
                                                      3)

    # TODO: check that `buffer_size=None` works
    dataset = dataset.prefetch(None)

    return dataset, seed_ph


import glob, os


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


data_dir = "D:\\data\\fashion_image_retrieval\\deep_fashion\\In-shop Clothes Retrieval Benchmark\\tfrecord"
files = glob.glob(os.path.join(data_dir, "*_train_*tfrecord"))
print(files)
dataset = tf.data.TFRecordDataset(files)
dataset = dataset.map(train_pre_process)

# data_dir = "c:\source/tensorflow-image-classification-framework/mnist"
# files = glob.glob(os.path.join(data_dir, "*_validation_*tfrecord"))
files = glob.glob(os.path.join(data_dir, "*_train_*tfrecord"))
print(files)
dataset2 = tf.data.TFRecordDataset(files)
dataset2 = dataset2.map(train_pre_process)
# dataset = tf.data.Dataset.zip((dataset, dataset2))

dataset, seed_ph = balanced_train_input_fn(dataset)
dataset2, seed_ph2 = balanced_train_input_fn(dataset2)
# dataset = tf.data.Dataset.zip((dataset, dataset2))
# iterator = dataset.make_one_shot_iterator()
# iterator = dataset.make_initializable_iterator()
iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                           dataset.output_shapes)
img, index_labels = iterator.get_next()
training_init_op = iterator.make_initializer(dataset)
# iterator2 = dataset2.make_one_shot_iterator()
# img2, index_labels2 = iterator2.get_next()

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)

sess.run(training_init_op)
# sess.run(iterator.initializer)
print(1)
for i in range(1):
    j = 0
    while True:
        try:
            # print(2)
            # images, lll, lll2 = sess.run([img, index_labels, index_labels2])
            # print(3)
            # im = Image.fromarray(images[0].astype('uint8'))

            # print(lll, lll2)
            # sys.exit()

            # images, lll = sess.run([img, index_labels],feed_dict={seed_ph:1, seed_ph2:1})
            images, lll = sess.run([img, index_labels])
            # print(images[1])
            print(lll)
            images, lll = sess.run([img, index_labels])
            # print(images[1])
            print(lll)
            sys.exit()
            images, lll = sess.run([img, index_labels])
            print(images[1])
            print(lll[1])

            images, lll = sess.run([img, index_labels])
            print(images[1])
            print(lll[1])

            images, lll = sess.run([img, index_labels])
            print(images[1])
            print(lll[1])
            # images, lll = sess.run([img, index_labels], feed_dict={seed_ph: 2, seed_ph2: 2})
            # images, lll = sess.run([img, index_labels])
            # print(images[1])
            # print(lll[1])
            # print(len(lll))
            # print(len(images))
            # print(images[0].shape)
            # print(images[1].shape)
            # print(lll[0].shape)
            # print(lll[1].shape)
            # print(lll)
            # images, lll = sess.run([img, index_labels])
            # im = Image.fromarray(images[0].astype('uint8'))
            # print(lll)
            # im.show()
            # if j == 0:
            # im.save("%d.jpg" % i)
            break
            # j += 1
            # if i == 1:
            #     break
        except tf.errors.OutOfRangeError:
            print("EXPCE")
            break
sys.exit()
