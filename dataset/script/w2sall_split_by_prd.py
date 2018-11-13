import os, glob, shutil

image_dir = "D:/data/fashion/image_retrieval/warehouse2shopall/images_only_product"
dataset_dir = "D:/data/fashion/image_retrieval/warehouse2shopall/dataset_only_product"

dirs = glob.glob(os.path.join(image_dir, "*"))
to_di = len(dirs)
for di, d in enumerate(dirs):
    imgs = glob.glob(os.path.join(d, "*.jpg"))
    tot = len(imgs)
    for i, im in enumerate(imgs):
        print("dirs: %d/%d, files %d/%d, %s" % (di, to_di, i, tot, im))
        file_name = os.path.basename(im)
        fn_split = file_name.split("_")
        prd_no = fn_split[1]
        if not prd_no.isdigit():
            print("file name error")
            continue

        output_dir = os.path.join(dataset_dir, prd_no)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        os.rename(im, os.path.join(output_dir, os.path.basename(im)))
