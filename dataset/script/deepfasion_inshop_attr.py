import glob, os
import pandas as pd
import json

output_path = "D:/data/fashion/image_retrieval/deep_fashion/In-shop Clothes Retrieval Benchmark/Anno"
split_df = pd.read_csv(
    "D:/data/fashion/image_retrieval/deep_fashion/In-shop Clothes Retrieval Benchmark/Eval/list_eval_partition_pd.txt",
    delim_whitespace=True)

attr_df = pd.read_csv(
    "D:/data/fashion/image_retrieval/deep_fashion/In-shop Clothes Retrieval Benchmark/Anno/list_attr_items_pd.txt",
    delim_whitespace=True)
attr_path = "D:/data/fashion/image_retrieval/deep_fashion/In-shop Clothes Retrieval Benchmark/Anno/list_attr_items_pd.txt"
attr_map = {}
for line in open(attr_path):
    line_split = line.rstrip("/n").split(" ")
    attr_map[line_split[0]] = [int(v) for v in line_split[1:]]

attr_file_map = {}

total = len(split_df)
i=0
for row in split_df.iterrows():
    print("train", i, total)
    i+=1
    file_name = os.path.basename(row[1]["image_name"])
    prd_no = os.path.basename(os.path.dirname(row[1]["image_name"]))
    key = "%s-%s" % (prd_no, file_name)
    attr_file_map[key] = attr_map[prd_no]

json.dump(attr_file_map, open(os.path.join(output_path, "attr_file_map.json"), mode="w+"))
