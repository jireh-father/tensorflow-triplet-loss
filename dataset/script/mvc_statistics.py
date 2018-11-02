import pandas as pd
import os, glob
import numpy as np
import sys

image_dir = "D:/data/fashion/image_retrieval/mvc/images/images_resize"
output_dir = image_dir + "_test"
df = pd.read_json("d:/data/fashion/image_retrieval/mvc/mvc_info.json")

test_index_dir = "D:/data/fashion/image_retrieval/mvc/test_files"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
if not os.path.isdir(test_index_dir):
    os.makedirs(test_index_dir)
# print(df.columns)
#
# print(df.productId.nunique())

# category_stat = df.groupby(['subCategory2'])["subCategory2"].count() * 0.5
# category_stat = df.groupby(['subCategory2'])["subCategory2"].count() * 0.5
category_prd_stat = df.groupby('subCategory2').productId.nunique() * 0.5
# print(df.groupby(['subCategory2'])["subCategory2"].count())
# sys.exit()
# nono! count by productId
category_prd_stat = category_prd_stat.astype(int)
aggregation_functions = {'subCategory2': 'first', 'productId': 'first'}
df_new = df.groupby(df['productId']).aggregate(aggregation_functions)
print(category_prd_stat)
for key in category_prd_stat.keys():
    print(key)
    indices = df_new[df_new.subCategory2 == key].sample(int(category_prd_stat[key]), random_state=1).productId.values
    target_df = df[df.productId.isin(indices)]
    urls = target_df.image_url_4x.values
    np.array(urls).dump(os.path.join(test_index_dir, "%s_test_urls.npy" % (key.strip('"'))))
    total = len(urls)
    for i, url in enumerate(urls):
        print(i, total)
        filename = os.path.basename(url)
        file_path = os.path.join(image_dir, filename)
        output_path = os.path.join(output_dir, filename)
        if os.path.isfile(output_path):
            print("skip")
            continue
        if os.path.isfile(file_path):
            os.rename(file_path, os.path.join(output_dir, filename))
            print(file_path)
        else:
            print("error!!!", file_path)
            sys.exit()
