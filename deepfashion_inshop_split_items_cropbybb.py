import pandas as pd
import math, os, shutil
from PIL import Image


def permutation(n, r):
    return int(math.factorial(n) / math.factorial(n - r))


df = pd.read_csv("D:\data\deep_fashion\In-shop Clothes Retrieval Benchmark\Eval\\list_eval_partition_pd.txt",
                 delim_whitespace=True)
df_bb = pd.read_csv("D:\data\deep_fashion\In-shop Clothes Retrieval Benchmark\Anno/list_bbox_inshop_pd.txt",
                    delim_whitespace=True)
df = df.set_index('image_name')
df_bb = df_bb.set_index('image_name')
df = pd.concat([df, df_bb], axis=1, join='inner')


def copy_files(df_tmp, image_dir, output_parent_dir, is_high_res=False):
    total = len(df_tmp)
    i = 0
    image_size = 224
    for index, row in df_tmp.iterrows():
        print(i, total)
        i += 1
        item_id = row["item_id"]
        image_name = index
        output_dir = os.path.join(output_parent_dir, item_id)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        if is_high_res:
            file_path = os.path.join(image_dir, "img_highres", "/".join(image_name.split("/")[1:]))
        else:
            file_path = os.path.join(image_dir, image_name)

        img = Image.open(file_path)
        copy_path = os.path.join(output_dir, os.path.basename(image_name))
        img = img.crop((int(row["x_1"]), int(row["y_1"]), int(row["x_2"]), int(row["y_2"])))
        old_size = img.size  # old_size[0] is in (width, height) format
        desired_size = image_size
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        img = img.resize(new_size, Image.ANTIALIAS)

        new_im = Image.new("RGB", (desired_size, desired_size), "black")
        new_im.paste(img, ((desired_size - new_size[0]) // 2,
                           (desired_size - new_size[1]) // 2))

        new_im.save(copy_path)


image_dir = "D:/data/deep_fashion/In-shop Clothes Retrieval Benchmark/Img"
df_tmp = df[df.evaluation_status == "train"]
output_parent_dir = image_dir + "_train_crop"
copy_files(df_tmp, image_dir, output_parent_dir)

df_tmp = df[(df.evaluation_status == "gallery") | (df.evaluation_status == "query")]
output_parent_dir = image_dir + "_test_crop"
copy_files(df_tmp, image_dir, output_parent_dir)

df_tmp = df[df.evaluation_status == "gallery"]
output_parent_dir = image_dir + "_index_crop"
copy_files(df_tmp, image_dir, output_parent_dir)

df_tmp = df[(df.evaluation_status == "query")]
output_parent_dir = image_dir + "_query_crop"
copy_files(df_tmp, image_dir, output_parent_dir)