import os, sys
import numpy as np

url_file = "D:/data/fashion/image_retrieval/street2shop/photos/photos.txt"
assert os.path.isfile(url_file)

f = open(url_file)
lines = f.readlines()
f.close()
extract_patterns = ["http://images.bloo", "http://images.asos"]
extracted_list = []
tot = len(lines)
for i, line in enumerate(lines):
    print(i, tot)
    line_split = line.rstrip('\n').split(",")

    url = line_split[1]
    if len(line_split) > 2:
        url += ("," + (",".join(line_split[2:])))
    if url[:18] in extract_patterns:
        extracted_list.append(line)

np.array(extracted_list).dump("D:/data/fashion/image_retrieval/street2shop/photos/remain_photos.npy")
