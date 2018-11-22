import glob, os

dir_list = glob.glob("D:/data/fashion/image_retrieval/deep_fashion/In-shop Clothes Retrieval Benchmark/split/*")

for d in dir_list:
    prd_list = glob.glob(os.path.join(d, "*"))
    for prd in prd_list:
        prd_no = os.path.basename(prd)
        files = glob.glob(os.path.join(prd, "*.jpg"))
        for i, fn in enumerate(files):
            print(i, len(files))
            os.rename(fn, os.path.join(prd, "%s-%s" % (prd_no, os.path.basename(fn))))
