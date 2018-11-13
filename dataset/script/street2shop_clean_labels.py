import os, glob, json
import numpy as np
import json

from PIL import Image

data_dir = "D:/data/fashion/image_retrieval/street2shop/meta/json"
output_data_dir = "D:/data/fashion/image_retrieval/street2shop/meta/json_new"
image_dir = "D:/data/fashion/image_retrieval/street2shop/images_resize"

if not os.path.isdir(output_data_dir):
    os.makedirs(output_data_dir)

files = glob.glob(os.path.join(data_dir, "*.json"))
files.reverse()
category_list = {}
bbox_list = {}
file_cnt = len(files)
for i, f in enumerate(files):
    print(f)
    category = os.path.splitext(os.path.basename(f))[0].split("_")[-1]
    data = json.load(open(f))
    clean_list = []
    total = len(data)
    for j, item in enumerate(data):
        print(i, file_cnt, j, total)
        file_name = "%s.jpg" % str(item["photo"])
        print(file_name)
        if os.path.isfile(os.path.join(image_dir, file_name)):
            clean_list.append(item)
            file_key = "%s/%s" % (str(item["product"]), file_name)
            if file_key not in category_list:
                category_list[file_key] = []
            category_list[file_key].append(category)
            if "bbox" in item:
                w, h = Image.open(os.path.join(image_dir, file_name)).size
                tmp_bbox = {"xmin": float(item["bbox"]["left"]) / float(w),
                            "xmax": float(item["bbox"]["left"] + item["bbox"]["width"]) / float(w),
                            "ymin": float(item["bbox"]["top"]) / float(h),
                            "ymax": float(item["bbox"]["top"] + item["bbox"]["height"]) / float(h)}
                if file_key not in bbox_list:
                    bbox_list[file_key] = []
                bbox_list[file_key].append(tmp_bbox)

    np.array(clean_list).dump(os.path.join(data_dir, os.path.splitext(os.path.basename(f))[0] + ".npy"))
json.dump(category_list, open(os.path.join(output_data_dir, "category.json"), mode="w+"))
json.dump(bbox_list, open(os.path.join(output_data_dir, "bbox.json"), mode="w+"))
