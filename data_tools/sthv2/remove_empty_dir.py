import os
import shutil

base_path = "/ssd01/data/sthv2/rawframes"

assert os.path.isdir(base_path)

cnt = 0

for dir_name in os.listdir(base_path):
    cur_dir = os.path.join(base_path, dir_name)
    if os.path.isdir(cur_dir) and len(os.listdir(cur_dir)) <= 15:
        # os.removedirs(cur_dir)
        shutil.rmtree(cur_dir)
        print(cur_dir)
        cnt += 1

print(cnt)
