import pandas as pd
import os
import cv2


def size(img):
    im = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    # height, width, number of channels in image
    return im.shape[0], im.shape[1]


csv_path = "D:/data/fashion/image_retrieval/cafe24multi/AreaInfo.csv"
image_dir = ""
df = pd.read_csv(csv_path)
fn_path = "D:/data/aipd/picked_dataset/file_names2.json"
import json

file_names = json.load(open(fn_path))
ok = {}
no = []
for row in df.iterrows():

    file_name = os.path.basename(row[1]["이미지 파일명"])
    if file_name in file_names:
        t_w = int(row[1]["이미지 Width"])
        t_h = int(row[1]["이미지 Height"])
        img = file_names[file_name]
        h, w = size(img)
        print(t_w, w, t_h, h)
        if t_w != w or t_h != h:
            no.append([img, t_w, t_h])
        # print("ok", file_name)
        ok[file_name] = True
    # else:
    #     # print(i[1])
    #     print(file_name)
    #     no[file_name] = True
print(no)

