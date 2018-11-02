import os, glob, shutil
from PIL import Image

image_dir = "D:/data/fashion/image_retrieval/warehouse2shop/3_shop"
moving_dir = "D:/data/fashion/image_retrieval/warehouse2shop/3_shop_moved"
if not os.path.isdir(moving_dir):
    os.makedirs(moving_dir)

limit_size = 300

dirs = glob.glob(os.path.join(image_dir, "*"))
for d in dirs:
    if not os.path.isdir(d):
        continue
    files = glob.glob(os.path.join(d, "*.jpg"))
    for f in files:
        im = Image.open(f)
        if im.size[0] <= limit_size or im.size[1] <= limit_size:
            print(f)
            im.close()
            im = None
            shutil.move(f, os.path.join(moving_dir, os.path.basename(f)))
