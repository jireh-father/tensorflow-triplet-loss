import pandas as pd
import os

csv_path = "D:/data/fashion/image_retrieval/cafe24multi_bb/AreaInfo.csv"
image_dir = ""
df = pd.read_csv(csv_path)
fn_path = "D:/data/aipd/picked_dataset/file_names.json"
import json

malls = ['boom2004', 'dabainsang', 'gqsharp', 'huns8402', 'khyelyun', 'mbaby', 'qlqkfnql', 'romi00a', 'shoplagirl',
         'skfo900815', 'skiinny', 'soozys', 'tiag86', 'yjk2924']

file_names = json.load(open(fn_path))
ok = {}
no = {}
in_cnt = 0
out_cnt = 0
for i in df["이미지 파일명"].iteritems():
    file_name = os.path.basename(i[1])
    if file_name.split("_")[0] not in malls:
        no[file_name] = True
        continue
    if file_name in file_names:
        # print("ok", file_name)
        ok[file_name] = True
    else:
        # print(i[1])
        print(file_name)
        no[file_name] = True
print(in_cnt, out_cnt)
print(len(ok), len(no))
