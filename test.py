import util
import glob

files = glob.glob("E:/data/adience_kaggle/test/*.tfrecord")
files += glob.glob("E:\data/adience_kaggle/faces/*.tfrecord")
print(files)
ma = util.create_label_map(files)
print(ma)



def ct(s, a):
    t = 0
    for r in a:
        if r[1].startswith(s):
            t += 1
    return t

def gr(a, pc):
    d = {}
    for r in a:
        if r[1][:pc] not in d:
            d[r[1][:pc]] = []
        d[r[1][:pc]].append(r[1])
    return d


def gr2(a, pc):
    d = {}
    for r in a:
        if r[1][:pc] not in d:
            d[r[1][:pc]] = []
        d[r[1][:pc]].append([r])
    return d


def grc(a, pc):
    d = {}
    for r in a:
        if r[1][:pc] not in d:
            d[r[1][:pc]] = 0
        d[r[1][:pc]] += 1
    return d


def grc2(a, pc):
    d = {}
    for r in a:
        r = r.rstrip('\n').split(",")
        if r[1][:pc] not in d:
            d[r[1][:pc]] = 0
        d[r[1][:pc]] += 1
    return d