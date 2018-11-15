import glob, os, shutil
import random

SEED = 1

image_dir = "D:/data/cafe24_clothing_category/images_split_cate_and_prd"
output_dir = "D:/data/cafe24_clothing_category/images_split_by_malls"
train_dir = "D:/data/cafe24_clothing_category/images_split_by_malls/train"
test_dir = "D:/data/cafe24_clothing_category/images_split_by_malls/test"
skip_categories = ["accessories", "bag", "cosmetics", "glasses", "gloves", "hat", "inner_wear",
                   "muffler", "socks", "swim_wear", "waistband", "watch", "long_skirt", "mini_skirt", "pants", "shoes",
                   "shorts", "skirt"]
cate_dirs = glob.glob(os.path.join(image_dir, "*"))
d_tot = len(cate_dirs)
for i, cate_dir in enumerate(cate_dirs):
    category = os.path.basename(cate_dir)
    if category in skip_categories:
        print("skip")
        continue
    mall_prd_dirs = glob.glob(os.path.join(cate_dir, "*"))
    malls = set()
    for mall_prd_dir in mall_prd_dirs:
        malls.add(os.path.basename(mall_prd_dir).split("_")[0])
    for mall_id in malls:
        mall_dirs = glob.glob(os.path.join(cate_dir, "%s_*" % mall_id))
        mall_prd_cnt = len(mall_dirs)
        test_cnt = int(mall_prd_cnt * 0.1)
        random.seed(SEED)
        random.shuffle(mall_dirs)
        mall_test_dirs = mall_dirs[:test_cnt]
        mall_train_dirs = mall_dirs[test_cnt:]
        te_tot = len(mall_test_dirs)
        for j, mall_test_dir in enumerate(mall_test_dirs):
            print("dir %d/%d, file %d/%d" % (i, d_tot, j, te_tot))
            mall_prd_dir = os.path.basename(mall_test_dir)

            files = glob.glob(os.path.join(mall_test_dir, "*.jpg"))
            output_dir = os.path.join(test_dir, mall_id)
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            for file_path in files:
                if not os.path.exists(os.path.join(output_dir, os.path.basename(file_path))):
                    shutil.copy(file_path, os.path.join(output_dir, os.path.basename(file_path)))

        tr_tot = len(mall_train_dirs)
        for j, mall_train_dir in enumerate(mall_train_dirs):
            print("dir %d/%d, file %d/%d" % (i, d_tot, j, tr_tot))
            mall_prd_dir = os.path.basename(mall_train_dir)
            files = glob.glob(os.path.join(mall_train_dir, "*.jpg"))
            output_dir = os.path.join(train_dir, mall_id)
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            for file_path in files:
                if not os.path.exists(os.path.join(output_dir, os.path.basename(file_path))):
                    shutil.copy(file_path, os.path.join(output_dir, os.path.basename(file_path)))
