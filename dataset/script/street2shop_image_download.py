from urllib.parse import urlparse
import urllib.request
import urllib.error
import os, sys
from PIL import Image
import glob
import numpy as np
import argparse
import requests

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default='/home/data/street2shop/images',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_file', default='/home/data/street2shop/url/1_photos.npy',
                    help="Directory containing the dataset")

args = parser.parse_args()

if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)
assert os.path.isdir(args.output_dir)

assert os.path.isfile(args.data_file)

download_error = []
image_error = []
convert_error = []
success = 0
lines = np.load(args.data_file)
for line_raw in lines:
    line = line_raw.rstrip('\n').split(",")
    o = urlparse(line[1])
    file_name = str(int(line[0]))
    url = line[1]
    ext = os.path.splitext(o.path)[1][1:].lower()
    if ext == "jpeg":
        ext = "jpg"
    output_file_path = os.path.join(args.output_dir, ("%s.%s" % (file_name, ext)))
    if ext == '':
        output_file_path += "jpg"
    print(url)

    try:
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, output_file_path)
    except:
        download_error.append(line)
        print("failed to download the url %s" % url)
        continue

    try:
        im = Image.open(output_file_path)
    except:
        print("failed to open image file %s", output_file_path)
        image_error.append(line)
        os.remove(output_file_path)
        continue

    if im.format != 'JPEG':
        old_path = output_file_path
        os.remove(old_path)
        output_file_path = "%s.%s" % (file_name, "jpg")
        try:
            im = im.convert('RGB')
            im.save(output_file_path, "JPEG", quality=100)
        except:
            convert_error.append(line)
            print("failed to convert to jpg image %s", output_file_path)
            continue

    success += 1

import uuid

key = uuid.uuid4()
np.array(download_error).dump("download_error_%.npy" % key)
np.array(image_error).dump("image_error_%.npy" % key)
np.array(convert_error).dump("convert_error_%.npy" % key)

print("download_error", len(download_error))
print("image_error", len(image_error))
print("convert_error", len(convert_error))
