import glob, os, shutil

data_dir = "D:/data/fashion/fashion_style14_v1/FashionStyle14_v1"

train_file = os.path.join(data_dir, "train.csv")
val_file = os.path.join(data_dir, "val.csv")
test_file = os.path.join(data_dir, "test.csv")


def cp(fn, phase):
    skip = 0
    tf = open(fn, encoding="utf-8")
    for i, line in enumerate(tf):

        line = line.rstrip('\n')
        file_path = os.path.join(data_dir, line)
        print("%s, %d, %s" % (phase, i, file_path))
        if not os.path.isfile(file_path):
            print(skip)
            skip += 1
            continue
        cp_path = os.path.join(data_dir, phase, os.path.basename(os.path.dirname(line)))
        if not os.path.isdir(cp_path):
            os.makedirs(cp_path)
        shutil.copy(file_path, cp_path)
    return skip


skip = 0
skip += cp(train_file, "train")
skip += cp(val_file, "val")
skip += cp(test_file, "test")
print(skip)
