from flask import Flask, render_template, redirect, url_for, request
import tensorflow as tf
from preprocessing import preprocessing_factory
from model import nets_factory
import numpy as np
import os, uuid, glob
import util
import faiss

fl = tf.app.flags
image_preprocessing_fn = 'inception'
model_name = 'inception_resnet_v2'
checkpoint_dir = '/home/source/tensorflow-triplet-loss/experiments/dfi_inception_resnet_v2_hard'
index_tfrecord_pattern = '/home/data/deepfashion-inshop/*index*.tfrecord'
faiss_gpu_no = '1'
image_size = 299
embedding_size = 128
max_top_k = 50
F = fl.FLAGS

UPLOAD_DIR = 'static/upload'
SEARCHED_DIR = 'static/result'
ALLOWED_EXTENSIONS = ['jpg']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

image_preprocessing_fn = preprocessing_factory.get_preprocessing(F.image_preprocessing_fn, is_training=False)


def _dataset_preprocess(example_proto):
    features = {"image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
                "image/class/label": tf.FixedLenFeature((), tf.int64, default_value=0),
                "image/class/name": tf.FixedLenFeature((), tf.string, default_value="")
                }

    parsed_features = tf.parse_single_example(example_proto, features)
    image = tf.image.decode_jpeg(parsed_features["image/encoded"], 3)

    image = image_preprocessing_fn(image, F.image_size, F.image_size)

    label = parsed_features["image/class/label"]
    label_name = parsed_features["image/class/name"]
    return image, label, label_name


def _image_file_preprocess(filename):
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = image_preprocessing_fn(image, F.image_size, F.image_size)

    return image


file_names = tf.placeholder(tf.string, shape=[None], name="file_names")
dataset = tf.data.Dataset.from_tensor_slices(file_names)
dataset = dataset.map(_image_file_preprocess)
dataset = dataset.batch(1)
iterator = dataset.make_initializable_iterator()
images_op = iterator.get_next()

model_f = nets_factory.get_network_fn(F.model_name, F.embedding_size, is_training=False)
with tf.variable_scope('model'):
    embeddings_op, _ = model_f(images_op)
saver = tf.train.Saver(tf.global_variables())

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)
sess.run(tf.global_variables_initializer())
print(tf.train.latest_checkpoint(F.checkpoint_dir))
saver.restore(sess, tf.train.latest_checkpoint(F.checkpoint_dir))

index_embeddings = np.load(os.path.join(F.checkpoint_dir, "index_embeddings.npy")).astype(np.float32)
index_labels = np.load(os.path.join(F.checkpoint_dir, "index_labels.npy"))

db_index = faiss.IndexFlatL2(int(F.embedding_size))
if F.faiss_gpu_no != "":
    db_index = faiss.index_cpu_to_all_gpus(  # build the index
        db_index
    )
max_top_k = int(F.max_top_k)
db_index.add(index_embeddings)  # add vectors to the index
print(db_index.ntotal)


@app.route("/")
def main_page():
    return render_template("image_retrieval.html")


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if not os.path.isdir(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            file_name = file.filename
            file_name = "%s_%s" % (uuid.uuid4(), file_name)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
            file.save(file_path)
            return redirect(url_for('.search', file_name=file_name))

    return redirect(request.url)


@app.route("/search")
def search():
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], request.args['file_name'])
    sess.run(iterator.initializer, feed_dict={file_names: [file_path]})
    embeddings = sess.run(embeddings_op)
    print(embeddings[0])
    print(embeddings[0].shape)
    print(file_path)
    print("start search!")
    searched_dist_list, searched_idx_list = db_index.search(embeddings, max_top_k)
    print("end search!")

    result_images = util.get_images_by_indices(glob.glob(F.index_tfrecord_pattern), list(searched_idx_list[0]),
                                               return_array=False)
    sub_dir = os.path.basename(file_path).split("_")[0]
    result_file_names = []
    for i, result_image in enumerate(result_images):
        result_file_name = os.path.join(SEARCHED_DIR, sub_dir, "%3d.jpg" % i)
        result_file_names.append(result_file_name)
        result_image.save(result_file_name)
    return render_template("search.html", query_file_name=request.args['file_name'], result_sub_dir=sub_dir,
                           result_file_names=result_file_names)


def allowed_file(file_name):
    return '.' in file_name and \
           file_name.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == "__main__":
    app.run()
