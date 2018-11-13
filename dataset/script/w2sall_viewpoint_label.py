import json
import os, glob

image_dir = "D:/data/fashion/image_retrieval/warehouse2shopall/dataset"
label_path = "D:/data/fashion/image_retrieval/warehouse2shopall/viewpoint_label.json"

label_map = {}
dirs = glob.glob(os.path.join(image_dir, "*"))
for d in dirs:
    viewpoint = os.path.basename(d)
    images = glob.glob(os.path.join(d, "*.jpg"))
    tot = len(images)
    for i, im in enumerate(images):
        print(i, tot)
        file_name = os.path.basename(im)
        label_map[file_name] = viewpoint
json.dump(label_map, open(label_path, mode="w+"))
