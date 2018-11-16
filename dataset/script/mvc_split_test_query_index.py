import glob, os, shutil
import random

SEED = 1

test_dir = "D:/data/fashion/image_retrieval/mvc/images/test"
query_dir = "D:/data/fashion/image_retrieval/mvc/images/tfrecord_test/query"
index_dir = "D:/data/fashion/image_retrieval/mvc/images/tfrecord_test/index"

dirs = glob.glob(os.path.join(test_dir, "*"))
d_tot = len(dirs)
for i, d in enumerate(dirs):
    prd = os.path.basename(d)
    files = glob.glob(os.path.join(d, "*.jpg"))
    f_tot = len(files)
    query_cnt = int(f_tot * 0.5)
    random.seed(SEED)
    random.shuffle(files)
    query_files = files[:query_cnt]
    index_files = files[query_cnt:]
    query_tot = len(query_files)
    for j, query_file in enumerate(query_files):
        print("dir %d/%d, file %d/%d" % (i, d_tot, j, query_tot))
        output_dir = os.path.join(query_dir, prd)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        shutil.copy(query_file, os.path.join(output_dir, os.path.basename(query_file)))

    index_tot = len(index_files)
    for j, index_file in enumerate(index_files):
        print("dir %d/%d, file %d/%d" % (i, d_tot, j, index_tot))
        output_dir = os.path.join(index_dir, prd)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        shutil.copy(index_file, os.path.join(output_dir, os.path.basename(index_file)))
