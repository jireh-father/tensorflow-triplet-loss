import glob, os, shutil

image_dir = "D:/data/aipd/crop"
output_dir = "D:/data/fashion/image_retrieval/cafe24multi/images_mall_prd"
mall_id_list = os.listdir(image_dir)

malls_cnt = len(mall_id_list)
for i, mall_id in enumerate(mall_id_list):
    mall_dir = os.path.join(image_dir, mall_id)
    if not os.path.isdir(mall_dir):
        print("not dir")
        continue
    image_path_list = glob.glob(os.path.join(mall_dir, "*.jpg"))
    images_cnt = len(image_path_list)
    for j, image_path in enumerate(image_path_list):
        print("mall %d/%d, image %d/%d : %s" % (i, malls_cnt, j, images_cnt, image_path))
        image_fn = os.path.basename(image_path)
        image_fn_split = image_fn.split("_")
        if image_fn_split[0] != mall_id:
            print("skip not mall id")
            continue
        if not image_fn_split[1].isdigit():
            print("skip not prd no")
            continue

        tmp_output_dir = os.path.join(output_dir, mall_id, image_fn_split[1])
        if os.path.isfile(os.path.join(tmp_output_dir, os.path.basename(image_path))):
            print("skip: already exists")
            continue
        if not os.path.isdir(tmp_output_dir):
            os.makedirs(tmp_output_dir)
        shutil.copy(image_path, tmp_output_dir)
