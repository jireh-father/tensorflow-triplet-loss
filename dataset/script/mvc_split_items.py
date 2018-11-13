import pandas as pd
import os, glob
import numpy as np
import sys
import shutil

train_dir = "D:/data/fashion/image_retrieval/mvc/images/images_resize"
test_dir = "D:/data/fashion/image_retrieval/mvc/images/images_resize_test"
train_output = os.path.join(os.path.dirname(train_dir), "train")
test_output = os.path.join(os.path.dirname(test_dir), "test")
df = pd.read_json("d:/data/fashion/image_retrieval/mvc/mvc_info.json")

if not os.path.isdir(train_output):
    os.makedirs(train_output)
if not os.path.isdir(test_output):
    os.makedirs(test_output)
tot = len(df)
for i, row in enumerate(df.iterrows()):
    print(i, tot)
    product_id = str(row[1].productId)
    file_name = os.path.basename(row[1].image_url_4x)
    train_path = os.path.join(train_dir, file_name)
    test_path = os.path.join(test_dir, file_name)
    from_path = None
    if os.path.isfile(train_path):
        output_dir = os.path.join(train_output, product_id)
        from_path = train_path
    else:
        output_dir = os.path.join(test_output, product_id)
        from_path = test_path
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    shutil.copy(from_path, output_dir)
