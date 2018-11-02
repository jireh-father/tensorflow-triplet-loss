import os, glob, shutil

image_dirs = ["F:/data/main_image/meetkmi0/sliced/desc_image", "F:/data/main_image/meetkmi0/sliced/main_image"]
output_dir = "D:/data/fashion/image_retrieval/warehouse2shop/3_shop"

assert len(image_dirs) > 0

for image_dir in image_dirs:
    assert os.path.isdir(image_dir)
    files = glob.glob(os.path.join(image_dir, "*.jpg"))
    for file in files:
        print(file)
        file_split = os.path.basename(file).split("_")
        if len(file_split) < 3:
            print("file exception", file)
            continue
        prd_no = file_split[1]
        if not prd_no.isdigit():
            print("prd no exception", file)
            continue
        new_output_path = os.path.join(output_dir, prd_no)
        if not os.path.isdir(new_output_path):
            os.makedirs(new_output_path)
        shutil.copy(file, os.path.join(new_output_path, os.path.basename(file)))
