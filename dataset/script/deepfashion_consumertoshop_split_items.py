import pandas as pd
import math, os, shutil


def permutation(n, r):
    return int(math.factorial(n) / math.factorial(n - r))


df = pd.read_csv("D:/data/fashion/image_retrieval/deep_fashion/consumer-to-shop/Eval/list_eval_partition_pd.txt",
                 delim_whitespace=True)


# total_cnt = df.item_id.nunique()
# train_cnt = df[df.evaluation_status == "train"].item_id.nunique()
# test_cnt = df[df.evaluation_status == "test"].item_id.nunique()
# val_cnt = df[df.evaluation_status == "val"].item_id.nunique()
# print(total_cnt)
# print(train_cnt)
# print(val_cnt)
# print(test_cnt)
# 
# consumer_cnt = df.image_pair_name_1.nunique()
# shop_cnt = df.image_pair_name_2.nunique()
# print(consumer_cnt)
# print(shop_cnt)
# 
# train_consumer_cnt = df[df.evaluation_status == "train"].image_pair_name_1.nunique()
# train_shop_cnt = df[df.evaluation_status == "train"].image_pair_name_2.nunique()
# train_total_cnt = train_consumer_cnt + train_shop_cnt
# print(train_consumer_cnt)
# print(train_shop_cnt)
# print(train_total_cnt)
# 
# val_consumer_cnt = df[df.evaluation_status == "val"].image_pair_name_1.nunique()
# val_shop_cnt = df[df.evaluation_status == "val"].image_pair_name_2.nunique()
# val_total_cnt = val_consumer_cnt + val_shop_cnt
# print(val_consumer_cnt)
# print(val_shop_cnt)
# print(val_total_cnt)
# 
# test_consumer_cnt = df[df.evaluation_status == "test"].image_pair_name_1.nunique()
# test_shop_cnt = df[df.evaluation_status == "test"].image_pair_name_2.nunique()
# test_total_cnt = test_consumer_cnt + test_shop_cnt
# print(test_consumer_cnt)
# print(test_shop_cnt)
# print(test_total_cnt)
# 
# # avg img count each item : 6.47
# a = df[df.evaluation_status == "train"].groupby(['item_id']).agg(['count']).mean()
# # max img count : 162
# b = df[df.evaluation_status == "train"].groupby(['item_id']).agg(['count']).max()
# c = df[df.evaluation_status == "train"].groupby(['item_id']).agg(['count']).min()
# print(a, b, c)
# 


def copy_files_for_train(df_tmp, image_dir, output_parent_dir):
    total = len(df_tmp)
    i = 0
    for index, row in df_tmp.iterrows():
        print(i, total)
        i += 1
        item_id = row["item_id"]
        consumer = "img_highres" + "/" + ("/".join(row["image_pair_name_1"].split("/")[1:]))
        shop = "img_highres" + "/" + ("/".join(row["image_pair_name_2"].split("/")[1:]))
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
        # consumer = row["image_pair_name_1"]
        # shop = row["image_pair_name_2"]
        consumer = "img_highres" + "/" + ("/".join(row["image_pair_name_1"].split("/")[1:]))
        shop = "img_highres" + "/" + ("/".join(row["image_pair_name_2"].split("/")[1:]))
        output_dir_query = os.path.join(output_parent_dir + "_query", item_id)
        output_dir_index = os.path.join(output_parent_dir + "_index", item_id)
        if not os.path.isdir(output_dir_query):
            os.makedirs(output_dir_query)
        if not os.path.isdir(output_dir_index):
            os.makedirs(output_dir_index)
        file_path = os.path.join(image_dir, consumer)
        if not os.path.isfile(os.path.join(output_dir_query, os.path.basename(consumer))):
            shutil.copy(file_path, os.path.join(output_dir_query, os.path.basename(consumer)))

        file_path = os.path.join(image_dir, shop)
        if not os.path.isfile(os.path.join(output_dir_index, os.path.basename(shop))):
            shutil.copy(file_path, os.path.join(output_dir_index, os.path.basename(shop)))


image_dir = "D:/data/fashion/image_retrieval/deep_fashion/consumer-to-shop/Img"
# df_tmp = df[df.evaluation_status == "train"]
# output_parent_dir = image_dir + "_train"
# copy_files_for_train(df_tmp, image_dir, output_parent_dir)

df_tmp = df[df.evaluation_status == "train"]
output_parent_dir = image_dir + "_train"
copy_files(df_tmp, image_dir, output_parent_dir)

df_tmp = df[df.evaluation_status == "val"]
output_parent_dir = image_dir + "_val"
copy_files(df_tmp, image_dir, output_parent_dir)

df_tmp = df[(df.evaluation_status == "test")]
output_parent_dir = image_dir + "_test"
copy_files(df_tmp, image_dir, output_parent_dir)
