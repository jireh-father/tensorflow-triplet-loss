import glob, os, shutil

test_q_dir = "D:/data/fashion/image_retrieval/street2shop/dataset/test/query"
test_i_dir = "D:/data/fashion/image_retrieval/street2shop/dataset/test/index"

q_dirs = glob.glob(os.path.join(test_q_dir, "*"))
for q_d in q_dirs:
    prd_no = os.path.basename(q_d)
    if not os.path.isdir(os.path.join(test_i_dir, prd_no)):
        print("remove %s" % prd_no)
        shutil.rmtree(q_d)
