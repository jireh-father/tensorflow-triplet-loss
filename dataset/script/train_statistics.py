import glob, os
import numpy as np

image_dir = "D:/data/fashion/image_retrieval/mvc/images/train"

dirs = glob.glob(os.path.join(image_dir, "*"))

cnt_list = []
for d in dirs:
    files = glob.glob(os.path.join(d, "*.jpg"))
    cnt_list.append(len(files))

cnt_list = np.array(cnt_list)

print(cnt_list.max())
print(cnt_list.min())
print(cnt_list.mean())
