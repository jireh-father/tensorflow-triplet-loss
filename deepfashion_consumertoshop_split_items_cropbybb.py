import pandas as pd
import math, os, shutil
from PIL import Image


def permutation(n, r):
    return int(math.factorial(n) / math.factorial(n - r))


df = pd.read_csv("D:\data\deep_fashion\consumer-to-shop\Eval\\list_eval_partition_pd.txt",
                 delim_whitespace=True)
df_bb = pd.read_csv("D:\data\deep_fashion\consumer-to-shop\Anno/list_bbox_consumer2shop_pd.txt",
                    delim_whitespace=True)


def copy_files_query_index(df_tmp, df_bb, image_dir, output_parent_dir, is_high_res=False, image_size=224):
    total = len(df_tmp)

    i = 0
    for index, row in df_tmp.iterrows():
        print(i, total)
        i += 1
        item_id = row["item_id"]
        consumer = row["image_pair_name_1"]
        shop = row["image_pair_name_2"]
        output_dir_query = os.path.join(output_parent_dir + "_query", item_id)
        output_dir_index = os.path.join(output_parent_dir + "_index", item_id)
        if not os.path.isdir(output_dir_query):
            os.makedirs(output_dir_query)
        if not os.path.isdir(output_dir_index):
            os.makedirs(output_dir_index)
        file_path = os.path.join(image_dir, consumer)
        crop_save(file_path, output_dir_query, consumer, df_bb[df_bb.image_name == consumer], image_size)

        file_path = os.path.join(image_dir, shop)
        crop_save(file_path, output_dir_index, shop, df_bb[df_bb.image_name == shop], image_size)


def crop_save(file_path, output_dir, image_name, bb, image_size=224):
    img = Image.open(file_path)
    copy_path = os.path.join(output_dir, os.path.basename(image_name))
    img = img.crop((int(bb["x_1"]), int(bb["y_1"]), int(bb["x_2"]), int(bb["y_2"])))
    old_size = img.size  # old_size[0] is in (width, height) format
    desired_size = image_size
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = img.resize(new_size, Image.ANTIALIAS)

    new_im = Image.new("RGB", (desired_size, desired_size), "black")
    new_im.paste(img, ((desired_size - new_size[0]) // 2,
                       (desired_size - new_size[1]) // 2))

    new_im.save(copy_path)


def copy_files(df_tmp, image_dir, output_parent_dir, is_high_res=False, image_size=224):
    total = len(df_tmp)
    i = 0

    for index, row in df_tmp.iterrows():
        print(i, total)
        i += 1
        item_id = row["item_id"]
        consumer = row["image_pair_name_1"]
        shop = row["image_pair_name_2"]
        output_dir = os.path.join(output_parent_dir, item_id)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(image_dir, consumer)
        crop_save(file_path, output_dir, consumer, df_bb[df_bb.image_name == consumer], image_size)

        file_path = os.path.join(image_dir, shop)
        crop_save(file_path, output_dir, shop, df_bb[df_bb.image_name == shop], image_size)


image_size = 224
image_dir = "D:/data/deep_fashion/consumer-to-shop/Img"
# df_tmp = df[df.evaluation_status == "train"]
# output_parent_dir = image_dir + "_train_crop"
# copy_files(df_tmp, image_dir, output_parent_dir, image_size=image_size)

df_tmp = df[df.evaluation_status == "val"]
output_parent_dir = image_dir + "_val_crop"
copy_files_query_index(df_tmp, df_bb, image_dir, output_parent_dir, image_size=image_size)
#
# df_tmp = df[(df.evaluation_status == "test")]
# output_parent_dir = image_dir + "_test_crop"
# copy_files_query_index(df_tmp, df_bb, image_dir, output_parent_dir, image_size=image_size)
#
# df_tmp = df[df.evaluation_status == "train"]
# output_parent_dir = image_dir + "_train_crop"
# copy_files_query_index(df_tmp, df_bb, image_dir, output_parent_dir, image_size=image_size)
