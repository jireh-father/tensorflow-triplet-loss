import os, shutil, glob

input_list = ["D:/data/fashion/image_retrieval/images_for_tfrecord/street2shop/train_query",
              "D:/data/fashion/image_retrieval/images_for_tfrecord/street2shop/train_index"]
output_path = "D:/data/fashion/image_retrieval/images_for_tfrecord/street2shop/train"

for idx, input_path in enumerate(input_list):
    dirs = glob.glob(os.path.join(input_path, "*"))
    total_dir = len(dirs)
    for i, d in enumerate(dirs):
        images = glob.glob(os.path.join(d, "*.jpg"))
        total_images = len(images)
        for j, im in enumerate(images):
            print("target %d/%d, dirs %d/%d, images %d/%d" % (idx, len(input_list), i, total_dir, j, total_images))
            output_dir = os.path.join(output_path, os.path.basename(d))
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            shutil.copy(im, output_dir)
