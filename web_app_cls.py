from flask import Flask, render_template, redirect, url_for, request
import tensorflow as tf
from preprocessing import preprocessing_factory
from model import nets_factory
import os, uuid
from grad_cam_plus_plus import GradCamPlusPlus
import cv2
from PIL import Image

preprocessing_name = 'inception'
model_name = 'inception_resnet_v2'
checkpoint_dir = 'D:\pretrained\\fs14_ir_ft_image_classification'
index_tfrecord_pattern = '/home/data/deepfashion-inshop/*index*.tfrecord'
image_size = 299
num_classes = 14
gpu_no = "0"

label_map = {0: "conservative",
             1: "dressy",
             2: "ethnic",
             3: "fairy",
             4: "feminine",
             5: "gal",
             6: "girlish",
             7: "kireime-casual",
             8: "lolita",
             9: "mode",
             10: "natural",
             11: "retro",
             12: "rock",
             13: "street"}

UPLOAD_DIR = 'static/upload'
CAM_DIR = 'static/cam'
ALLOWED_EXTENSIONS = ['jpg']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no

image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name, is_training=False)


def _image_file_preprocess(filename):
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = image_preprocessing_fn(image, image_size, image_size)

    return image


file_names = tf.placeholder(tf.string, shape=[None], name="file_names")
dataset = tf.data.Dataset.from_tensor_slices(file_names)
dataset = dataset.map(_image_file_preprocess)
dataset = dataset.batch(1)
iterator = dataset.make_initializable_iterator()
images_op = iterator.get_next()

model_f = nets_factory.get_network_fn(model_name, num_classes, is_training=False)
logits, end_points = model_f(images_op)
result_prob_op = tf.nn.softmax(logits)

cam = GradCamPlusPlus(logits, end_points["Conv2d_7b_1x1"], images_op)

saver = tf.train.Saver(tf.global_variables())

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)
sess.run(tf.global_variables_initializer())
saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))


@app.route("/")
def main_page():
    return render_template("fashion_style_classification.html")


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
            return redirect(url_for('.classification', file_name=file_name))

    return redirect(request.url)


@app.route("/classification")
def classification():
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], request.args['file_name'])
    sess.run(iterator.initializer, feed_dict={file_names: [file_path]})
    result_probs, images_input, logits_input = sess.run([result_prob_op, images_op, logits])
    result_idx = result_probs[0].argmax()

    cam_imgs, class_indices = cam.create_cam_imgs(sess, images_input, logits_input)
    heatmap = cam.convert_cam_2_heatmap(cam_imgs[0][0])
    overlay_img = cam.overlay_heatmap(images_input[0], heatmap)
    cam_image = overlay_img[..., ::-1]
    print(cam_image.shape)
    cam_filename = "%s.jpg" % uuid.uuid4()
    if not os.path.isdir("static/cam"):
        os.makedirs("static/cam")
    Image.fromarray(cam_image).save(os.path.join("static/cam", cam_filename))
    print("finished to summary cam")

    return render_template("classification.html", query_file_name=request.args['file_name'], result_idx=result_idx,
                           result_probs=result_probs[0], label_map=label_map, num_classes=num_classes,
                           cam_filename=cam_filename)


def allowed_file(file_name):
    return '.' in file_name and \
           file_name.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == "__main__":
    app.run()
