import glob, os

path_list = [
    "D:/data/fashion/image_retrieval/deep_fashion/consumer-to-shop/Img_test_index",
    "D:/data/fashion/image_retrieval/deep_fashion/consumer-to-shop/Img_test_query",
    "D:/data/fashion/image_retrieval/deep_fashion/consumer-to-shop/Img_train_index",
    "D:/data/fashion/image_retrieval/deep_fashion/consumer-to-shop/Img_train_query",
    "D:/data/fashion/image_retrieval/deep_fashion/consumer-to-shop/Img_val_index",
    "D:/data/fashion/image_retrieval/deep_fashion/consumer-to-shop/Img_val_query",
    "D:/data/fashion/image_retrieval/street2shop/tfrecord_images/train/index",
    "D:/data/fashion/image_retrieval/street2shop/tfrecord_images/train/query",
    "D:/data/fashion/image_retrieval/street2shop/tfrecord_images/test/index",
    "D:/data/fashion/image_retrieval/street2shop/tfrecord_images/test/query",
    "D:/data/fashion/image_retrieval/warehouse2shopall/test/index",
    "D:/data/fashion/image_retrieval/warehouse2shopall/test/query",
]
check_exts = ["jpeg", 'png', 'tif', 'tiff', 'bmp', 'gif', 'jpg']

for path in path_list:
    counts = {"jpeg": 0, 'png': 0, 'tif': 0, 'tiff': 0, 'bmp': 0, 'gif': 0, 'jpg': 0}
    dirs = glob.glob(os.path.join(path, "*"))
    for d in dirs:
        files = glob.glob(os.path.join(d, "*"))
        for fn in files:
            if os.path.splitext(fn)[1][1:] in check_exts:
                counts[os.path.splitext(fn)[1][1:]] += 1

    print(path, counts)
