import pandas as pd
import math, os, shutil


def permutation(n, r):
    return int(math.factorial(n) / math.factorial(n - r))


df = pd.read_csv(
    "D:/data/fashion/image_retrieval/deep_fashion/In-shop Clothes Retrieval Benchmark/Eval/list_eval_partition_pd.txt",
    delim_whitespace=True)

train_cnt = df[df.evaluation_status == "train"].item_id.nunique()
test_cnt = df[df.evaluation_status == "query"].item_id.nunique()
print(train_cnt)
print(test_cnt)

# avg img count each item : 6.47
a = df[df.evaluation_status == "train"].groupby(['item_id']).agg(['count']).mean()
# max img count : 162
b = df[df.evaluation_status == "train"].groupby(['item_id']).agg(['count']).max()
c = df[df.evaluation_status == "train"].groupby(['item_id']).agg(['count']).min()
print(a, b, c)
print("std", df[df.evaluation_status == "train"].groupby(['item_id']).agg(['count']).std())
# item_id group by count
df2 = df[df.evaluation_status == "train"].groupby(['item_id'])["item_id"].count()
total = 0
for n in df2:
    if n < 2:
        continue
    if n == 2:
        total += 1
        continue
    total += permutation(n, 2)


def copy_files(df_tmp, image_dir, output_parent_dir):
    total = len(df_tmp)
    i = 0
    for index, row in df_tmp.iterrows():
        print(i, total)
        i += 1
        item_id = row["item_id"]
        image_name = row["image_name"]
        output_dir = os.path.join(output_parent_dir, item_id)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(image_dir, image_name)
        shutil.copy(file_path, os.path.join(output_dir, os.path.basename(image_name)))


image_dir = "D:/data/deep_fashion/In-shop Clothes Retrieval Benchmark/Img"
df_tmp = df[df.evaluation_status == "train"]
output_parent_dir = image_dir + "_train"
copy_files(df_tmp, image_dir, output_parent_dir)

# df_tmp = df[(df.evaluation_status == "gallery") | (df.evaluation_status == "query")]
# output_parent_dir = image_dir + "_test"
# copy_files(df_tmp, image_dir, output_parent_dir)

df_tmp = df[df.evaluation_status == "gallery"]
output_parent_dir = image_dir + "_index"
copy_files(df_tmp, image_dir, output_parent_dir)

df_tmp = df[(df.evaluation_status == "query")]
output_parent_dir = image_dir + "_query"
copy_files(df_tmp, image_dir, output_parent_dir)
