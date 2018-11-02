import glob, os

image_dir = "F:/data/main_image/meetkmi0/sliced/desc_image"
remove_patterns = ["*wasing-*", ]
for pattern in remove_patterns:
    files = glob.glob(os.path.join(image_dir, pattern))
    for file in files:
        os.remove(file)
