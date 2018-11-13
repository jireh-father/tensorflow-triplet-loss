import os, glob

# image_path = "D:/data/fashion/image_retrieval/cafe24product/dataset_train"
#
# prd_list = glob.glob(os.path.join(image_path, "*"))
#
# cnt_list = {}
# for prd_dir in prd_list:
#     cnt = len(glob.glob(os.path.join(prd_dir, "*.jpg")))
#     if cnt not in cnt_list:
#         cnt_list[cnt] = 0
#     cnt_list[cnt] += 1
# print(cnt_list)

image_path = "D:/data/fashion/image_retrieval/cafe24product/dataset_test/query"

prd_list = glob.glob(os.path.join(image_path, "*"))

cnt_list = {}
for prd_dir in prd_list:
    cnt = len(glob.glob(os.path.join(prd_dir, "*.jpg")))
    if cnt not in cnt_list:
        cnt_list[cnt] = 0
    cnt_list[cnt] += 1
print(cnt_list)
