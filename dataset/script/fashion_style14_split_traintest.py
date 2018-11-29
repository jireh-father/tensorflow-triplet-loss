import glob, os, shutil
import random
import json

data_dir = "D:/data/fashion/fashion_style14_v1/FashionStyle14_v1/dataset"
train_output = os.path.join(os.path.dirname(data_dir), "train")
test_output = os.path.join(os.path.dirname(data_dir), "test")
train_ratio = 0.7
errors = {"train": [], "test": []}
style_dirs = glob.glob(os.path.join(data_dir, "*"))
random.seed(0)
for style_idx, style_dir in enumerate(style_dirs):
    image_files = glob.glob(os.path.join(style_dir, "*.jpg"))
    random.shuffle(image_files)
    train_cnt = int(float(len(image_files)) * train_ratio)
    train_files = image_files[:train_cnt]
    test_files = image_files[train_cnt:]
    for i, train_file in enumerate(train_files):
        print(train_file, style_idx, len(style_dirs), "train", i, len(train_files))
        output_dir = os.path.join(train_output, os.path.basename(style_dir))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        if os.path.isfile(os.path.join(output_dir, os.path.basename(train_file))):
            print("skip")
            continue
        try:
            shutil.copy(train_file, output_dir)
        except:
            errors["train"].append(train_file)
    for i, test_file in enumerate(test_files):
        print(style_idx, len(style_dirs), "test", i, len(test_files))
        output_dir = os.path.join(test_output, os.path.basename(style_dir))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        if os.path.isfile(os.path.join(output_dir, os.path.basename(test_file))):
            print("skip")
            continue
        try:
            shutil.copy(test_file, output_dir)
        except:
            errors["test"].append(test_file)
json.dump(errors, open(os.path.join(os.path.dirname(data_dir), "error.json"), mode="w+"))
