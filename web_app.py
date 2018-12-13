from flask import Flask, render_template, redirect, url_for, request
import tensorflow as tf
from preprocessing import preprocessing_factory
from model import nets_factory
import numpy as np
from numba import cuda
import os, uuid, glob
import util

import faiss

checkpoint_dir = './experiments/'
UPLOAD_DIR = 'static/upload'
SEARCHED_DIR = 'static/result'
ALLOWED_EXTENSIONS = ['jpg']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR

cp_dirs = glob.glob(os.path.join(checkpoint_dir, "*"))
model_list = [os.path.basename(cp_d) for cp_d in cp_dirs if os.path.isdir(cp_d)]
loaded_model = None

sess = None
iterator = None
embeddings_op = None
db_index = None
file_names = None

dataset_pattern = None
max_top_k = 20


def _load_model(model_name="inception_resnet_v2", preprocessing_name="inception", faiss_gpu_no=0, image_size=299,
                embedding_size=128, num_preprocessing_threads=4, use_old_model=False, cp_dir=None):
    global sess
    global iterator
    global embeddings_op
    global db_index
    global file_names

    if sess is not None:
        sess.close()
        tf.reset_default_graph()
        cuda.select_device(0)
        cuda.close()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name, is_training=False)

    def _image_file_preprocess(filename):
        image_string = tf.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = image_preprocessing_fn(image, image_size, image_size)

        return image

    file_names = tf.placeholder(tf.string, shape=[None], name="file_names")
    dataset = tf.data.Dataset.from_tensor_slices(file_names)
    dataset = dataset.map(_image_file_preprocess, num_parallel_calls=num_preprocessing_threads)
    dataset = dataset.batch(1)
    iterator = dataset.make_initializable_iterator()
    images_op = iterator.get_next()

    model_f = nets_factory.get_network_fn(model_name, embedding_size, is_training=False)
    if use_old_model:
        with tf.variable_scope('model'):
            embeddings_op, _ = model_f(images_op)
    else:
        embeddings_op, _ = model_f(images_op)
    saver = tf.train.Saver(tf.global_variables())

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())

    saver.restore(sess, tf.train.latest_checkpoint(cp_dir))

    index_embeddings = np.load(os.path.join(cp_dir, "index_embeddings.npy")).astype(np.float32)

    db_index = faiss.IndexFlatL2(int(embedding_size))
    if faiss_gpu_no != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = faiss_gpu_no
        db_index = faiss.index_cpu_to_all_gpus(  # build the index
            db_index
        )
    db_index.add(index_embeddings)  # add vectors to the index
    print(db_index.ntotal)


@app.route("/")
def main_page():
    global loaded_model
    global dataset_pattern
    global max_top_k

    preprocessing_name = 'inception'
    model_name = 'inception_resnet_v2'
    dataset_pattern = '/home/data/deepfashion-inshop/*index*.tfrecord'
    gpu_no = '0'
    image_size = 299
    embedding_size = 128
    max_top_k = 20
    num_preprocessing_threads = 4
    use_old_model = False

    print(request.args)

    if 'gpu_no' in request.args:
        gpu_no = request.args["gpu_no"]
    if 'model_name' in request.args:
        model_name = request.args["model_name"]
    if 'dataset_pattern' in request.args:
        dataset_pattern = request.args["dataset_pattern"]
    if 'image_size' in request.args:
        image_size = int(request.args["image_size"])
    if 'embedding_size' in request.args:
        embedding_size = int(request.args["embedding_size"])
    if 'max_top_k' in request.args:
        max_top_k = int(request.args["max_top_k"])
    if 'num_preprocessing_threads' in request.args:
        num_preprocessing_threads = int(request.args["num_preprocessing_threads"])
    if 'use_old_model' in request.args:
        use_old_model = True

    if 'model_path' in request.args and request.args["model_path"] != "none":
        try:
            _load_model(model_name, preprocessing_name, gpu_no, image_size, embedding_size, num_preprocessing_threads,
                        use_old_model, os.path.join(checkpoint_dir, request.args["model_path"]))
            loaded_model = request.args["model_path"]
        except:
            loaded_model = None
    return render_template("image_retrieval.html", model_list=model_list, loaded_model=loaded_model, gpu_no=gpu_no,
                           preprocessing_name=preprocessing_name, model_name=model_name,
                           dataset_pattern=dataset_pattern, image_size=image_size, embedding_size=embedding_size,
                           max_top_k=max_top_k, num_preprocessing_threads=num_preprocessing_threads,
                           use_old_model=use_old_model)


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

    result_images = util.get_images_by_indices(glob.glob(dataset_pattern), list(searched_idx_list[0]),
                                               return_array=False)
    sub_dir = os.path.basename(file_path).split("_")[0]
    result_file_names = []
    for i, result_image in enumerate(result_images):
        result_file_name = os.path.join(SEARCHED_DIR, sub_dir, "%d.jpg" % i)
        if not os.path.isdir(os.path.dirname(result_file_name)):
            os.makedirs(os.path.dirname(result_file_name))
        result_file_names.append(os.path.basename(result_file_name))
        result_image.save(result_file_name)
    return render_template("search.html", query_file_name=request.args['file_name'], result_sub_dir=sub_dir,
                           result_file_names=result_file_names, num_result_images=max_top_k,
                           result_dist_list=searched_dist_list[0])


def allowed_file(file_name):
    return '.' in file_name and \
           file_name.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == "__main__":
    app.run()
