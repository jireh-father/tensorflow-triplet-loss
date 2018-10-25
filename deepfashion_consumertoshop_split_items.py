import pandas as pd
import math, os, shutil


def permutation(n, r):
    return int(math.factorial(n) / math.factorial(n - r))


df = pd.read_csv("D:\data\deep_fashion\consumer-to-shop\Eval\\list_eval_partition_pd.txt",
                 delim_whitespace=True)


def copy_files_for_train(df_tmp, image_dir, output_parent_dir):
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
        shutil.copy(file_path, os.path.join(output_dir, os.path.basename(consumer)))

        file_path = os.path.join(image_dir, shop)
        shutil.copy(file_path, os.path.join(output_dir, os.path.basename(shop)))


def copy_files(df_tmp, image_dir, output_parent_dir):
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
        shutil.copy(file_path, os.path.join(output_dir_query, os.path.basename(consumer)))

        file_path = os.path.join(image_dir, shop)
        shutil.copy(file_path, os.path.join(output_dir_index, os.path.basename(shop)))


image_dir = "D:/data/deep_fashion/consumer-to-shop/Img"
df_tmp = df[df.evaluation_status == "train"]
output_parent_dir = image_dir + "_train"
copy_files_for_train(df_tmp, image_dir, output_parent_dir)

df_tmp = df[df.evaluation_status == "train"]
output_parent_dir = image_dir + "_train"
copy_files(df_tmp, image_dir, output_parent_dir)

df_tmp = df[df.evaluation_status == "val"]
output_parent_dir = image_dir + "_val"
copy_files(df_tmp, image_dir, output_parent_dir)

df_tmp = df[(df.evaluation_status == "test")]
output_parent_dir = image_dir + "_test"
copy_files(df_tmp, image_dir, output_parent_dir)
