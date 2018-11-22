import os, glob, shutil

image_dir = "D:/data/fashion/image_retrieval/cafe24product/dataset_train"

prd_dirs = glob.glob(os.path.join(image_dir, "*"))
total = 0
cnt_list = []
for d in prd_dirs:
    files = glob.glob(os.path.join(d, "*.jpg"))
    cnt_list.append(len(files))
import numpy as np

cnt_list = np.array(cnt_list)
print(cnt_list.min())
print(cnt_list.max())
print(cnt_list.mean())
print(cnt_list.std())
print("total images", total)
