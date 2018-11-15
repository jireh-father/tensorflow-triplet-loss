import glob, os, shutil

image_dir = "D:/data/cafe24_clothing_category/cate_cut_0411"
output_dir = "D:/data/cafe24_clothing_category/images_split_cate_and_prd"

dirs = glob.glob(os.path.join(image_dir, "*"))
d_tot = len(dirs)
for i, d in enumerate(dirs):
    category = os.path.basename(d)

    files = glob.glob(os.path.join(d, "*.jpg"))
    f_tot = len(files)
    for j, f in enumerate(files):
        print("dir %d/%d, file %d/%d" % (i, d_tot, j, f_tot))
        f_split = os.path.basename(f).split("_")
        if not f_split[2].isdigit():
            print("skip")
            continue
        prd = "%s_%s" % (f_split[0], f_split[2])
        output_path = os.path.join(output_dir, category, prd)
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        shutil.copy(f, output_path)
