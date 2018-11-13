import glob, os, shutil
import random

SEED = 1

image_dir = "D:/data/cafe24_clothing_category/images_split_cate_and_prd"
train_dir = "D:/data/cafe24_clothing_category/images_train"
test_dir = "D:/data/cafe24_clothing_category/images_test"

dirs = glob.glob(os.path.join(image_dir, "*"))
d_tot = len(dirs)
for i, d in enumerate(dirs):
    category = os.path.basename(d)
    prds = glob.glob(os.path.join(d, "*"))
    p_tot = len(prds)
    test_cnt = int(p_tot * 0.5)
    random.seed(SEED)
    random.shuffle(prds)
    test_dirs = prds[:test_cnt]
    train_dirs = prds[test_cnt:]
    te_tot = len(test_dirs)
    for j, test_d in enumerate(test_dirs):
        print("dir %d/%d, file %d/%d" % (i, d_tot, j, te_tot))
        pd = os.path.basename(test_d)
        files = glob.glob(os.path.join(test_d, "*.jpg"))
        output_dir = os.path.join(test_dir, pd)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        for f in files:
            shutil.copy(f, os.path.join(output_dir, os.path.basename(f)))

    tr_tot = len(train_dirs)
    for j, train_d in enumerate(train_dirs):
        print("dir %d/%d, file %d/%d" % (i, d_tot, j, tr_tot))
        pd = os.path.basename(train_d)
        files = glob.glob(os.path.join(train_d, "*.jpg"))
        output_dir = os.path.join(train_dir, pd)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        for f in files:
            shutil.copy(f, os.path.join(output_dir, os.path.basename(f)))
