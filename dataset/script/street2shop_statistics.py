import os, glob, shutil

image_dir = "D:/data/fashion/image_retrieval/street2shop/tfrecord_images/train"

dirs = glob.glob(os.path.join(image_dir, "*"))
total = 0
cnt_list = []
for d in dirs:
    categories = glob.glob(os.path.join(d, "*"))
    for cate in categories:
        files = glob.glob(os.path.join(cate, "*.jpg"))
        cnt_list.append(len(files))
import numpy as np

cnt_list = np.array(cnt_list)
print(cnt_list.min())
print(cnt_list.max())
print(cnt_list.mean())
print("total images", total)
