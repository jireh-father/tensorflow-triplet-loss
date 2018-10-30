from urllib.parse import urlparse
import urllib.request
import urllib.error
import os, sys
from PIL import Image
import glob
import numpy as np

output_dir = "/home/data/street2shop/images"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
assert os.path.isdir(output_dir)

data_dir = "/home/data/street2shop/url"
files = glob.glob(os.path.join(data_dir, "*.npy"))
assert len(files) > 0

download_error = []
image_error = []
convert_error = []
success = 0
for np_file in files:
    lines = np.load(np_file)
    for line_raw in lines:
        line = line_raw.rstrip('\n').split(",")
        o = urlparse(line[1])
        file_name = str(int(line[0]))
        url = line[1]
        ext = os.path.splitext(o.path)[1][1:].lower()
        if ext == "jpeg":
            ext = "jpg"
        output_file_path = os.path.join(output_dir, ("%s.%s" % (file_name, ext)))
        if ext == '':
            output_file_path += "jpg"

        print(url)

        try:
            urllib.request.urlretrieve(url, output_file_path)
        except urllib.error.HTTPError:
            download_error.append(url)
            print("failed to download the url %s" % url)
            continue

        try:
            im = Image.open(output_file_path)
        except OSError:
            print("failed to open image file %s", output_file_path)
            image_error.append(url)
            os.remove(output_file_path)
            continue

        if im.format != 'JPEG':
            old_path = output_file_path
            os.remove(old_path)
            output_file_path = "%s.%s" % (file_name, "jpg")
            try:
                im.save(output_file_path, "JPEG", quality=100)
            except:
                convert_error += 1
                image_error.append(url)
                print("failed to convert to jpg image %s", output_file_path)
                continue

        success += 1

print(download_error)
print(image_error)
print(convert_error)

print("download_error", len(download_error))
print("image_error", len(download_error))
print("convert_error", len(download_error))
