import glob, os
from PIL import Image
import pandas as pd
import json

bbox_file = "D:/data/fashion/image_retrieval/deep_fashion/consumer-to-shop/Anno/list_bbox_consumer2shop_pd.txt"
image_dir = "D:/data/fashion/image_retrieval/deep_fashion/consumer-to-shop/Img"
bbox_output = "D:/data/fashion/image_retrieval/deep_fashion/consumer-to-shop/Anno/bbox.json"
cloth_type_output = "D:/data/fashion/image_retrieval/deep_fashion/consumer-to-shop/Anno/cloth_type.json"
category_output = "D:/data/fashion/image_retrieval/deep_fashion/consumer-to-shop/Anno/category.json"
subcategory_output = "D:/data/fashion/image_retrieval/deep_fashion/consumer-to-shop/Anno/subcategory.json"

df = pd.read_csv(bbox_file, delim_whitespace=True)

clothes_type_map = {1: "upper-body", 2: "lower-body", 3: "full-body"}
bbox_map = {}
cloth_type_map = {}
category_map = {}
sub_category_map = {}
tot = len(df)
for i, row in enumerate(df.iterrows()):
    print(i, tot)
    file_name = row[1].image_name
    file_name_split = file_name.split("/")
    key_file_name = file_name_split[-2] + "-" + file_name_split[-1]
    xmin = row[1].x_1
    xmax = row[1].x_2
    ymin = row[1].y_1
    ymax = row[1].y_2
    cloth_type = row[1].clothes_type
    file_path = os.path.join(image_dir, file_name)
    assert os.path.isfile(file_path)
    w, h = Image.open(file_path).size
    tmp_bbox = {"xmin": float(xmin) / float(w),
                "xmax": float(xmax) / float(w),
                "ymin": float(ymin) / float(h),
                "ymax": float(ymax) / float(h)}
    bbox_map[key_file_name] = [tmp_bbox]
    cloth_type_map[key_file_name] = clothes_type_map[cloth_type]
    category_map[key_file_name] = file_name_split[1]
    sub_category_map[key_file_name] = file_name_split[2]

json.dump(bbox_map, open(bbox_output, mode="w+"))
json.dump(cloth_type_map, open(cloth_type_output, mode="w+"))
json.dump(category_map, open(category_output, mode="w+"))
json.dump(sub_category_map, open(subcategory_output, mode="w+"))
