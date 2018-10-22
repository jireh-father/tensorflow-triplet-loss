import pandas as pd
import math, os, shutil
from PIL import Image, ImageDraw

df = pd.read_csv("D:\data\deep_fashion\In-shop Clothes Retrieval Benchmark\Anno/list_bbox_inshop_pd.txt",
                 delim_whitespace=True)


def copy_files(df_tmp, image_dir, output_parent_dir, is_high_res=False):
    total = len(df_tmp)
    i = 0
    image_size = 224
    for index, row in df_tmp.iterrows():
        print(i, total)
        i += 1
        image_name = row["image_name"]
        file_path = os.path.join(image_dir, image_name)

        img = Image.open(file_path)
        draw = ImageDraw.Draw(img)
        draw.line([(int(row["x_1"]), int(row["y_1"])), (int(row["x_2"]), int(row["y_1"]))],
                  fill=(255, 0, 0), width=4)
        draw.line([(int(row["x_1"]), int(row["y_1"])), (int(row["x_1"]), int(row["y_2"]))],
                  fill=(255, 0, 0), width=4)
        draw.line([(int(row["x_2"]), int(row["y_1"])), (int(row["x_2"]), int(row["y_2"]))],
                  fill=(255, 0, 0), width=4)
        draw.line([(int(row["x_1"]), int(row["y_2"])), (int(row["x_2"]), int(row["y_2"]))],
                  fill=(255, 0, 0), width=4)
        copy_path = os.path.join(output_parent_dir, image_name)
        if not os.path.isdir(os.path.dirname(copy_path)):
            os.makedirs(os.path.dirname(copy_path))

        img.save(copy_path)


image_dir = "D:/data/deep_fashion/In-shop Clothes Retrieval Benchmark/Img"
output_parent_dir = image_dir + "_bb"
copy_files(df, image_dir, output_parent_dir)
