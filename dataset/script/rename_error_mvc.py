import os, glob

files = glob.glob("D:\data/fashion\image_retrieval\mvc\images\images/*.jpg")
total = len(files)
for i, f in enumerate(files):

    print(i, total)
    if os.path.basename(f).endswith("jpg.jpg"):
        os.rename(f, os.path.splitext(f)[0])
