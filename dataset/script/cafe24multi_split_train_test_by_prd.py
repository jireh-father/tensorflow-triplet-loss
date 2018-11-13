import glob, os, shutil, random

SEED = 1
image_dir = "D:/data/fashion/image_retrieval/cafe24multi/images_mall_prd"
output_dir = "D:/data/fashion/image_retrieval/cafe24multi/images_split_train_test"
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")
mall_id_list = os.listdir(image_dir)

malls_cnt = len(mall_id_list)
for i, mall_id in enumerate(mall_id_list):
    mall_dir = os.path.join(image_dir, mall_id)
    if not os.path.isdir(mall_dir):
        print("not dir")
        continue
    prds = glob.glob(os.path.join(mall_dir, "*"))
    prd_cnt = len(prds)

    p_tot = len(prds)
    test_cnt = int(prd_cnt * 0.5)
    random.seed(SEED)
    random.shuffle(prds)
    test_dirs = prds[:test_cnt]
    train_dirs = prds[test_cnt:]
    te_tot = len(test_dirs)
    for j, test_d in enumerate(test_dirs):
        print("dir %d/%d, file %d/%d" % (i, prd_cnt, j, te_tot))
        pd = os.path.basename(test_d)
        files = glob.glob(os.path.join(test_d, "*.jpg"))
        tmp_output_dir = os.path.join(test_dir, pd)
        if not os.path.isdir(tmp_output_dir):
            os.makedirs(tmp_output_dir)
        for f in files:
            shutil.copy(f, os.path.join(tmp_output_dir, os.path.basename(f)))

    tr_tot = len(train_dirs)
    for j, train_d in enumerate(train_dirs):
        print("dir %d/%d, file %d/%d" % (i, prd_cnt, j, tr_tot))
        pd = os.path.basename(train_d)
        files = glob.glob(os.path.join(train_d, "*.jpg"))
        tmp_output_dir = os.path.join(train_dir, pd)
        if not os.path.isdir(tmp_output_dir):
            os.makedirs(tmp_output_dir)
        for f in files:
            shutil.copy(f, os.path.join(tmp_output_dir, os.path.basename(f)))
