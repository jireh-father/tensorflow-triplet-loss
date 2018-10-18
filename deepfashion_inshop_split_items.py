import pandas as pd
import math, os, shutil


def permutation(n, r):
    return int(math.factorial(n) / math.factorial(n - r))


df = pd.read_csv("D:\data\deep_fashion\In-shop Clothes Retrieval Benchmark\Eval\\list_eval_partition_pd.txt",
                 delim_whitespace=True)
train_cnt = df[df.evaluation_status == "train"].item_id.nunique()
test_cnt = df[df.evaluation_status == "query"].item_id.nunique()

# avg img count each item : 6.47
df[df.evaluation_status == "train"].groupby(['item_id']).agg(['count']).mean()
# max img count : 162
df[df.evaluation_status == "train"].groupby(['item_id']).agg(['count']).max()
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
print(total)

# df = df[df.evaluation_status == "train"]

df = df[(df.evaluation_status == "gallery") | (df.evaluation_status == "query")]
total = len(df)
image_dir = "D:/data/deep_fashion/In-shop Clothes Retrieval Benchmark/Img"
output_parent_dir = image_dir + "_split_test"
for index, row in df.iterrows():

    print(index, total)
    item_id = row["item_id"]
    image_name = row["image_name"]
    output_dir = os.path.join(output_parent_dir, item_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(image_dir, image_name)
    shutil.copy(file_path, os.path.join(output_dir, os.path.basename(image_name)))
