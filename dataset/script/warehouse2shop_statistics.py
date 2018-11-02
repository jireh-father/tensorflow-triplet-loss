import os, glob, shutil

category_file = "D:/data/fashion/image_retrieval/warehouse2shop_for_train/category.txt"
image_dir = "D:/data/fashion/image_retrieval/warehouse2shop_for_train/train"

f = open(category_file)
lines = f.readlines()
category_list = {}
for line in lines:
    line_split = line.rstrip("/n").split(" ")
    category = line_split[0]
    num_prds = len(line_split) - 1
    category_list[category] = line_split[1:]
    print(category, num_prds)

dirs = glob.glob(os.path.join(image_dir, "*"))
total = 0
cnt_list = []
for d in dirs:

    files = glob.glob(os.path.join(d, "*.jpg"))
    cnt_list.append(len(files))
import numpy as np

cnt_list = np.array(cnt_list)
print(cnt_list.min())
print(cnt_list.max())
print(cnt_list.mean())
print("total images", total)

