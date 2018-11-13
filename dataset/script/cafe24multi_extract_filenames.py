import os, glob
import json

image_dirs = [
    "D:/data/aipd/crop", "D:/data/aipd/crop2",
    "D:/data/aipd/crop3", ]
output_path = "D:/data/aipd/picked_dataset/file_names2.json"
file_name_map = {}
for image_dir in image_dirs:
    dirs = glob.glob(os.path.join(image_dir, "*"))

    for d in dirs:
        if not os.path.isdir(d):
            continue
        for fn in glob.glob(os.path.join(d, "*.jpg")):
            file_name_map[os.path.basename(fn)] = fn
json.dump(file_name_map, open(output_path, mode="w+"))
