import os, sys
import numpy as np

url_file = "D:/data/fashion/image_retrieval/street2shop/photos/photos.txt"
assert os.path.isfile(url_file)

f = open(url_file)
lines = f.readlines()
f.close()
num_split = 12
total = len(lines)

num_split_lines = int(total / num_split)

for i in range(num_split):
    if i - 1 == num_split:
        chunk = np.array(lines[i * num_split_lines:])

    else:
        chunk = np.array(lines[i * num_split_lines:i * num_split_lines + num_split_lines])
    chunk.dump(os.path.join(os.path.dirname(url_file),
                            str(i) + "_" + os.path.splitext(os.path.basename(url_file))[0] + ".npy"))
