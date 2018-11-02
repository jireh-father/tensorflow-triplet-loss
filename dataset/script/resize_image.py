import numpy as np
from PIL import Image
import glob, os
import PIL.ImageOps

target_dir = "D:/data/fashion/image_retrieval/mvc/images/images"
output_dir = "D:/data/fashion/image_retrieval/mvc/images/images_resize"
assert os.path.isdir(target_dir)
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

image_filenames = glob.glob(os.path.join(target_dir, "*.jpg"))
image_size = 512
t = len(image_filenames)
errors = []
for i, fn in enumerate(image_filenames):
    print(t, i)
    output_path = os.path.join(output_dir, os.path.basename(fn))
    if os.path.isfile(output_path):
        print("skip")
        continue
    try:
        im = Image.open(fn)
        old_size = im.size  # old_size[0] is in (width, height) format
        desired_size = image_size
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        im = im.resize(new_size, Image.ANTIALIAS)
        # create a new image and paste the resized on it

        new_im = Image.new("RGB", (desired_size, desired_size), "white")
        new_im.paste(im, ((desired_size - new_size[0]) // 2,
                          (desired_size - new_size[1]) // 2))
        new_im.save(output_path)
        # img = np.asarray(new_im)
        new_im.close()
    except:
        print("error")
        errors.append(fn)

np.array(errors).dump("resize_error.npy")
