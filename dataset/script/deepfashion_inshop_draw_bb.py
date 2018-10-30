import pandas as pd
import math, os, shutil
from PIL import Image, ImageDraw

df = pd.read_csv("D:\data\deep_fashion\In-shop Clothes Retrieval Benchmark\Anno/list_bbox_inshop_pd.txt",
                 delim_whitespace=True)
df["width"] = df["x_2"] - df["x_1"]


def copy_files(df_tmp, image_dir, output_parent_dir, is_high_res=False):
    total = len(df_tmp)
    i = 0
    limit_width_percent = 0.3
    image_size = 224
    for index, row in df_tmp.iterrows():
        print(i, total)
        i += 1
        image_name = row["image_name"]
        file_path = os.path.join(image_dir, image_name)
        pose_type = row["pose_type"]
        img = Image.open(file_path)
        w = img.size[0]
        width_percent = row["width"] / w
        draw = ImageDraw.Draw(img)
        x_1 = int(row["x_1"])
        x_2 = int(row["x_2"])
        y_1 = int(row["y_1"])
        y_2 = int(row["y_2"])
        if pose_type == 2 and width_percent < limit_width_percent:
            if width_percent < 0.07:
                ww = row["width"] * 2
            else:
                ww = row["width"] * ((limit_width_percent + 0.1) - width_percent) * 2
            x_1e = row["x_1"] - ww
            x_2e = row["x_2"] + ww
            draw.line([(x_1e, y_1), (x_2e, y_1)], fill=(0, 255, 0), width=4)
            draw.line([(x_1e, y_1), (x_1e, y_2)], fill=(0, 255, 0), width=4)
            draw.line([(x_2e, y_1), (x_2e, y_2)], fill=(0, 255, 0), width=4)
            draw.line([(x_1e, y_2), (x_2e, y_2)], fill=(0, 255, 0), width=4)

        draw.line([(x_1, y_1), (x_2, y_1)], fill=(255, 0, 0), width=4)
        draw.line([(x_1, y_1), (x_1, y_2)], fill=(255, 0, 0), width=4)
        draw.line([(x_2, y_1), (x_2, y_2)], fill=(255, 0, 0), width=4)
        draw.line([(x_1, y_2), (x_2, y_2)], fill=(255, 0, 0), width=4)
        copy_path = os.path.join(output_parent_dir, image_name)
        if not os.path.isdir(os.path.dirname(copy_path)):
            os.makedirs(os.path.dirname(copy_path))

        img.save(copy_path)


image_dir = "D:/data/deep_fashion/In-shop Clothes Retrieval Benchmark/Img"
output_parent_dir = image_dir + "_bb_expanded"
copy_files(df, image_dir, output_parent_dir)
