import os, sys
import numpy as np
import json

url_file = "D:/data/fashion/image_retrieval/mvc/image_links.json"
assert os.path.isfile(url_file)
urls = json.load(open(url_file))
num_split = 12
total = len(urls)

num_split_lines = int(total / num_split)

for i in range(num_split):
    if i + 1 == num_split:
        chunk = np.array(urls[i * num_split_lines:])
    else:
        chunk = np.array(urls[i * num_split_lines:i * num_split_lines + num_split_lines])
    chunk.dump(os.path.join(os.path.dirname(url_file),
                            str(i) + "_" + os.path.splitext(os.path.basename(url_file))[0] + ".npy"))
