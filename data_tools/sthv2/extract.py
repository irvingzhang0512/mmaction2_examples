import os
import subprocess
from multiprocessing import Pool
from tqdm import tqdm

in_path = "/ssd01/data/sthv2/videos"
out_path = "/ssd01/data/sthv2/rawframes"
n_thread = 2
if not os.path.exists(out_path):
    os.mkdir(out_path)

ffmpeg_commands = "ffmpeg -i \"{}\" -q:v 0 \"{}/img_%05d.jpg\""
denseflow_commands = "denseflow '{}' -b=20 -s=0 -o='{}' -nw=0 -nh=0 -v"

denseflow = True


def extract_one_video(fname):
    dir_name = fname[:fname.rfind(".")]
    if not os.path.exists(os.path.join(out_path, dir_name)):
        os.mkdir(os.path.join(out_path, dir_name))
        # print("not exsits")
    else:
        # 存在文件就退出
        # print("exists")
        return

    if denseflow:
        cmd = denseflow_commands.format(os.path.join(in_path, fname), out_path)
    else:
        cmd = ffmpeg_commands.format(
            os.path.join(in_path, fname), os.path.join(out_path, dir_name))

    subprocess.call(
        cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    vid_list = os.listdir(in_path)
    vid_list.sort()
    p = Pool(n_thread)
    for _ in tqdm(
            p.imap_unordered(extract_one_video, vid_list),
            total=len(vid_list)):
        pass
    p.close()
    p.join()

    print('\n')
