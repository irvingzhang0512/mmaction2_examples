import os
import subprocess
from multiprocessing import Pool
from tqdm import tqdm

in_path = "/ssd/zhangyiyang/mmaction2/data/ava/videos"
out_path = "/ssd/zhangyiyang/mmaction2/data/ava/videos_15min"

n_thread = 5
if not os.path.exists(out_path):
    os.mkdir(out_path)

cmd_format = 'ffmpeg -ss 900 -t 901 -i "{}" -r 30 -strict experimental "{}"'


def cut_one_video(fname):
    if not os.path.exists(os.path.join(out_path, fname)):
        os.mkdir(os.path.join(out_path, fname))
        # print("not exsits")
    else:
        # 存在文件就退出
        # print("exists")
        return
    cmd = cmd_format.format(
        os.path.join(in_path, fname), os.path.join(out_path, fname))
    print(cmd)
    # subprocess.call(
    #     cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    vid_list = os.listdir(in_path)
    vid_list.sort()
    p = Pool(n_thread)
    for _ in tqdm(
            p.imap_unordered(cut_one_video, vid_list), total=len(vid_list)):
        pass
    p.close()
    p.join()

    print('\n')
