from urllib.parse import urlparse
import urllib.request
import urllib.error
import os, sys
from PIL import Image
import glob
import numpy as np
import argparse
import requests
import imghdr

parser = argparse.ArgumentParser()
# parser.add_argument('--output_dir', default='D:/data/fashion/image_retrieval/street2shop/images',
#                     help="Experiment directory containing params.json")
# parser.add_argument('--data_file', default='D:/data/fashion/image_retrieval/street2shop/photos/remain_photos.npy',
#                     help="Directory containing the dataset")


parser.add_argument('--output_dir', default='/home/data/street2shop/images',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_file',
                    default='/home/data/street2shop/url/download_error_19247f3b-41cf-4cc6-8339-bb4fc0db2fde.npy',
                    help="Directory containing the dataset")


def download(url, file_path):
    try:
        r = requests.get(url, timeout=10, allow_redirects=True, headers={
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36'})
        if r.status_code == 200:
            image_type = imghdr.what(None, r.content)

            if image_type is not None:
                with open(file_path, 'wb') as f:
                    f.write(r.content)
                    f.close()
            else:
                raise Exception
        else:
            raise Exception
    except:
        return False
    return True


args = parser.parse_args()

if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)
assert os.path.isdir(args.output_dir)

assert os.path.isfile(args.data_file)


def is_image(file_path):
    try:
        Image.open(file_path)
        return True
    except:
        return False


download_error = []
image_error = []
convert_error = []
success = 0
lines = np.load(args.data_file)
for line in lines:
    o = urlparse(line[1])
    file_name = str(int(line[0]))
    url = line[1]
    if len(line) > 2:
        url += ("," + (",".join(line[2:])))
    ext = os.path.splitext(o.path)[1][1:].lower()
    if ext == "jpeg":
        ext = "jpg"
    output_file_path = os.path.join(args.output_dir, ("%s.%s" % (file_name, ext)))
    if ext == '':
        output_file_path += "jpg"
    print(url)

    if ext == "jpg":
        if os.path.isfile(output_file_path) and is_image(output_file_path):
            print("skip")
            continue
    else:
        tmp_file_path = os.path.join(args.output_dir, ("%s.%s" % (file_name, 'jpg')))
        if os.path.isfile(tmp_file_path) and is_image(tmp_file_path):
            print("skip")
            continue

    # try:
    result = download(url, output_file_path)
    # opener = urllib.request.build_opener()
    # opener.addheaders = [('User-agent',
    #                       'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36')]
    # urllib.request.install_opener(opener)
    # urllib.request.urlretrieve(url, output_file_path)
    # except:
    if not result:
        import traceback

        traceback.print_exc()
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
np.array(download_error).dump("download_error_%s.npy" % key)
np.array(image_error).dump("image_error_%s.npy" % key)
np.array(convert_error).dump("convert_error_%s.npy" % key)

print("download_error", len(download_error))
print("image_error", len(image_error))
print("convert_error", len(convert_error))
