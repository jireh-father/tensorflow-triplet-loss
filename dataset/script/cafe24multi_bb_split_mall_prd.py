import pandas as pd
import os
from PIL import Image
import cv2
import json
import shutil


def size(img):
    im = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    # height, width, number of channels in image
    return im.shape[0], im.shape[1]


dataset_dir = "D:/data/fashion/image_retrieval/cafe24multi_bb/"
output_dir = "D:/data/fashion/image_retrieval/cafe24multi_bb/images_split_mall_prd"
csv_path = os.path.join(dataset_dir, "AreaInfo.csv")
fn_path = os.path.join(dataset_dir, "file_names2.json")
category_output_path = os.path.join(dataset_dir, "category.json")
subcategory_output_path = os.path.join(dataset_dir, "subcategory.json")
bbox_output_path = os.path.join(dataset_dir, "bbox.json")
polygon_path = os.path.join(dataset_dir, "polygon.json")

malls = ['boom2004', 'dabainsang', 'gqsharp', 'huns8402', 'khyelyun', 'mbaby', 'qlqkfnql', 'romi00a', 'shoplagirl',
         'skfo900815', 'skiinny', 'soozys', 'tiag86', 'yjk2924']

df = pd.read_csv(csv_path)
file_names = json.load(open(fn_path))
category = {}
subcategory = {}
polygon = {}
bbox = {}
total = len(df)
for i, row in enumerate(df.iterrows()):
    file_name = os.path.basename(row[1]["이미지 파일명"])
    print("row %d/%d : %s" % (i, total, file_name))
    split_file_name = file_name.split("_")
    if split_file_name[0] not in malls or not split_file_name[1].isdigit():
        print("skip: not mall or prd no")
        continue

    if file_name not in file_names:
        print("skip: the file doesn't exist.")
        continue
    original_width = int(row[1]["이미지 Width"])
    original_height = int(row[1]["이미지 Height"])
    local_img_path = file_names[file_name]
    print(local_img_path)
    local_height, local_width = size(local_img_path)

    if file_name not in category:
        category[file_name] = []
        subcategory[file_name] = []
        polygon[file_name] = []
        bbox[file_name] = []

    category[file_name].append(row[1]["카테고리명"])
    subcategory[file_name].append(row[1]["라벨명"])
    polygon[file_name].append(json.loads(row[1]["사각좌표"]))
    bbox[file_name].append(json.loads(row[1]["Annotation 좌표"]))

    tmp_output_dir = os.path.join(output_dir, split_file_name[0], split_file_name[1])
    if not os.path.isdir(tmp_output_dir):
        os.makedirs(tmp_output_dir)
    if original_width != local_width or original_height != local_height:
        im = Image.open(local_img_path)
        im = im.resize((original_width, original_height), Image.LANCZOS)
        im.save(os.path.join(tmp_output_dir, file_name))
    else:
        shutil.copy(local_img_path, tmp_output_dir)

json.dump(category, open(category_output_path, mode="w+"))
json.dump(subcategory, open(subcategory_output_path, mode="w+"))
json.dump(polygon, open(polygon_path, mode="w+"))
json.dump(bbox, open(bbox_output_path, mode="w+"))
