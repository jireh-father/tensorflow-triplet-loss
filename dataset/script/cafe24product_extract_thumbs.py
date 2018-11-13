import glob, os, shutil

image_dir = "D:/data/cafe24_clothing_category/cate_cut_0411"
output_dir = "D:/data/cafe24_clothing_category/original_thumb"

dirs = glob.glob(os.path.join(image_dir, "*"))
d_tot = len(dirs)
for i, d in enumerate(dirs):
    category = os.path.basename(d)
    files = glob.glob(os.path.join(d, "*_thumb.jpg"))
    f_tot = len(files)
    for j, f in enumerate(files):
        print("dir %d/%d, file %d/%d" % (i, d_tot, j, f_tot))
        output_path = os.path.join(output_dir, category)
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        os.rename(f, os.path.join(output_path, os.path.basename(f)))
