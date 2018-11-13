import glob, os
from PIL import Image

image_dir = "D:/data/fashion/image_retrieval/street2shop/images"

tif_images = glob.glob(os.path.join(image_dir, "*.tif"))
for tif in tif_images:
    im = Image.open(tif)
    im = im.convert("RGB")
    print(tif)
    print(os.path.splitext(tif)[0] + ".jpg")
    sys.exit()
    im.save(os.path.splitext(tif)[0] + ".jpg", "JPEG", quality=100)
