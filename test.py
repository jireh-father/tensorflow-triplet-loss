import util
import glob

files = glob.glob("E:/data/adience_kaggle/test/*.tfrecord")
files += glob.glob("E:\data/adience_kaggle/faces/*.tfrecord")
print(files)
ma = util.create_label_map(files)
print(ma)
