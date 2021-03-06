import tensorflow as tf


def _make_balanced_batched_dataset(datasets, num_classes, num_classes_per_batch,
                                   num_images_per_class):
    assert len(datasets) == num_classes, \
        "There should be {} datasets, got {}".format(num_classes, len(datasets))

    def generator(_):
        # Sample `num_classes_per_batch` classes for the batch
        sampled = tf.random_shuffle(tf.range(num_classes), 1)[:num_classes_per_batch]
        # Repeat each element `num_images_per_class` times
        batch_labels = tf.tile(tf.expand_dims(sampled, -1), [1, num_images_per_class])
        return tf.to_int64(tf.reshape(batch_labels, [-1]))

    selector = tf.contrib.data.Counter().map(generator)
    selector = selector.apply(tf.contrib.data.unbatch())

    dataset = tf.contrib.data.choose_from_datasets(datasets, selector)

    # Batch
    batch_size = num_classes_per_batch * num_images_per_class
    dataset = dataset.batch(batch_size)

    return dataset  # , seed_ph


def balanced_train_input_fn(dataset):
    print("go")
    datasets=[]
    for i in range(2):
        print(i, 2)
        datasets.append(dataset.filter(lambda img, lab: tf.equal(lab, i)))
    # datasets = [dataset.filter(lambda img, lab: tf.equal(lab, i)) for i in range(40989)]
    print("gogo")
    print(datasets)

    dataset = _make_balanced_batched_dataset(datasets,
                                             2,
                                             4,
                                             2)

    # TODO: check that `buffer_size=None` works
    dataset = dataset.prefetch(None)

    return dataset  # , seed_ph


import glob, os


def train_pre_process(example_proto):
    features = {"image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
                "image/class/label": tf.FixedLenFeature((), tf.int64, default_value=0),
                'image/height': tf.FixedLenFeature((), tf.int64, default_value=0),
                'image/width': tf.FixedLenFeature((), tf.int64, default_value=0)
                }
    parsed_features = tf.parse_single_example(example_proto, features)

    image = tf.image.decode_jpeg(parsed_features["image/encoded"], 1)
    # image = tf.cast(image, tf.float32)
    #
    # image = tf.expand_dims(image, 0)
    # image = tf.image.resize_bilinear(image, [224, 224], align_corners=False)
    # image = tf.squeeze(image, [0])
    #
    # image = tf.divide(image, 255.0)
    # image = tf.subtract(image, 0.5)
    # image = tf.multiply(image, 2.0)

    label = parsed_features["image/class/label"]
    # return parsed_features["image/encoded"], label
    return image, label


data_dir = "D:\data\\fashion\image_retrieval\cafe24product/tfrecord"
files = glob.glob(os.path.join(data_dir, "*_query_*tfrecord"))
print(files)
dataset = tf.data.TFRecordDataset(files)
dataset = dataset.map(train_pre_process)
sampled = tf.random_shuffle(tf.range(2), 1)[:2]
print(sampled)
dataset = dataset.filter(lambda img, lab: tf.equal(lab, i))
dataset = dataset.shuffle(100)

# data_dir = "c:\source/tensorflow-image-classification-framework/mnist"
# files = glob.glob(os.path.join(data_dir, "*_validation_*tfrecord"))
# files = glob.glob(os.path.join(data_dir, "*_train_*tfrecord"))
print(files)
dataset2 = tf.data.TFRecordDataset(files)
dataset2 = dataset2.map(train_pre_process)
# dataset = tf.data.Dataset.zip((dataset, dataset2))

dataset = balanced_train_input_fn(dataset)
# dataset2 = balanced_train_input_fn(dataset2)
# dataset = tf.data.Dataset.zip((dataset, dataset2))
iterator = dataset.make_one_shot_iterator()
# iterator = dataset.make_initializable_iterator()
# iterator = tf.data.Iterator.from_structure(dataset.output_types,
#                                            dataset.output_shapes)
img, index_labels = iterator.get_next()
# training_init_op = iterator.make_initializer(dataset)
# iterator2 = dataset2.make_one_shot_iterator()
# img2, index_labels2 = iterator2.get_next()

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)

# sess.run(training_init_op)
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
            lll = sess.run(index_labels)
            # print(images[1])
            print(lll)
            break
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
