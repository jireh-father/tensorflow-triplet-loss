import os, glob, shutil

image_path = "D:/data/fashion/image_retrieval/cafe24product/dataset_train"
query_dir = "D:/data/fashion/image_retrieval/cafe24product/dataset_test/query"
index_dir = "D:/data/fashion/image_retrieval/cafe24product/dataset_test/index"

query_set = set(os.path.basename(p) for p in glob.glob(os.path.join(query_dir, "*")))
print(query_set)
index_set = set(os.path.basename(p) for p in glob.glob(os.path.join(index_dir, "*")))
qi_diff = query_set - index_set
iq_diff = index_set - query_set
for prd_no in qi_diff:
    img_list = glob.glob(os.path.join(query_dir, prd_no, "*.jpg"))
    target_dir = os.path.join(image_path, prd_no)
    os.makedirs(target_dir)
    print(img_list)
    for im_path in img_list:
        print(im_path, os.path.join(target_dir, os.path.basename(im_path)))
        sys.exit()
        os.rename(im_path, os.path.join(target_dir, os.path.basename(im_path)))
