import glob, os, shutil
import numpy as np


def copy_file(image_dir, photo, output_dir):
    image_path = os.path.join(image_dir, "%s.jpg" % photo)
    if os.path.isfile(image_path):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        shutil.copy(image_path, output_dir)


meta_dir = "D:/data/fashion/image_retrieval/street2shop/meta/json"
category_path = "D:/data/fashion/image_retrieval/street2shop/meta/json/category.npy"
image_dir = "D:/data/fashion/image_retrieval/street2shop/images_resize"
dataset_dir = "D:/data/fashion/image_retrieval/street2shop/tfrecord_images"
train_query_dir = os.path.join(dataset_dir, "train", "query")
train_index_dir = os.path.join(dataset_dir, "train", "index")
test_query_dir = os.path.join(dataset_dir, "test", "query")
test_index_dir = os.path.join(dataset_dir, "test", "index")

retrieval_files = glob.glob(os.path.join(meta_dir, "retrieval_*.npy"))
train_files = glob.glob(os.path.join(meta_dir, "train_*.npy"))
test_files = glob.glob(os.path.join(meta_dir, "test_*.npy"))
retrieval_files.sort()
train_files.sort()
test_files.sort()

total = len(retrieval_files)
for i in range(len(retrieval_files)):
    index_file = retrieval_files[i]
    train_file = train_files[i]
    test_file = test_files[i]

    category = os.path.splitext(os.path.basename(index_file))[0].split("_")[1]
    print(category)
    indices = np.load(index_file)
    trains = np.load(train_file)
    tests = np.load(test_file)
    index_map = {}
    ind_total = len(indices)
    for j, ind in enumerate(indices):
        print("category %d/%d, file %d/%d" % (i, total, j, ind_total))
        photo = str(ind["photo"])
        product = str(ind["product"])
        if not product in index_map:
            index_map[product] = []
        index_map[product].append(photo)

    tr_total = len(trains)
    for j, tr in enumerate(trains):
        print("category %d/%d, file %d/%d" % (i, total, j, tr_total))
        photo = str(tr["photo"])
        product = str(tr["product"])
        if product not in index_map:
            print("skip: not in index for test")
            continue
        copy_file(image_dir, photo, os.path.join(train_query_dir, product))
        for index_photo in index_map[product]:
            copy_file(image_dir, index_photo, os.path.join(train_index_dir, product))

    te_total = len(tests)
    for j, te in enumerate(tests):
        print("category %d/%d, file %d/%d" % (i, total, j, te_total))
        photo = str(te["photo"])
        product = str(te["product"])
        if product not in index_map:
            print("skip: not in index for train")
            continue
        copy_file(image_dir, photo, os.path.join(test_query_dir, product))
        for index_photo in index_map[product]:
            copy_file(image_dir, index_photo, os.path.join(test_index_dir, product))
