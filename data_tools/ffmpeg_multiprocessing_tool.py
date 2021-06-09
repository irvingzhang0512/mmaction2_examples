import argparse
import os
import subprocess
from functools import partial
from multiprocessing import Pool

from tqdm import tqdm


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("in_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--num-processes", type=int, default=5)
    parser.add_argument("--command-type", type=str, default="resize_videos")
    parser.add_argument(
        "--shortside",
        type=int,
        default=320,
        help="args for 'resize_videos' command")
    parser.add_argument(
        "--skip",
        action="store_true",
        help="skip if the outout file/dir exists.")

    return parser.parse_args()


def _extract_frames(in_dir, out_dir, skip, file_name):
    """Extract frames from all videos in in_dir.

    1. Generate one dir for one input video.
    2. The frame file name format is `%05d.jpg`
    """
    base_name = file_name[:file_name.rfind(".")]
    cur_out_dir = os.path.join(out_dir, base_name)
    if skip and os.path.exists(cur_out_dir):
        print(f"{file_name} exists, skip...")
        return
    os.mkdir(cur_out_dir)
    command = (f'ffmpeg -i "{os.path.join(in_dir, file_name)}" -r 30 -q:v 1 '
               f'{cur_out_dir + "/img_%05d.jpg"}')

    # print(command)
    subprocess.call(
        command,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)


def _resize_videos(in_dir, out_dir, skip, shortside, file_name):
    """Resize all videos from in_dir and save results in out_dir.

    All viedos are resized with origin aspect ratio. And the shortside of
    the output video is determined by the method arg.
    """
    outfile_path = os.path.join(out_dir, file_name)
    if skip and os.path.exists(outfile_path):
        print(f"{file_name} exists, skip...")
        return
    command = " ".join([
        "ffmpeg", "-i",
        os.path.join(in_dir, file_name), "-vf",
        (f"\"scale=iw*{shortside}/'min(ih,iw)':ih*{shortside}/'min(iw,ih)',"
         f"pad=ceil(iw*{shortside}/'min(ih,iw)'/2)*2:"
         f"ceil(ih*{shortside}/'min(iw,ih)'/2)*2\""), outfile_path
    ])

    # print(command)
    subprocess.call(
        command,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)


def _cut_videos(in_dir, out_dir, skip, file_name):
    """Cut videos for ava dataset.

    Get sub video(15:00 - 30:00, aka 901 seconds long) from the origin one.
    """
    outfile_path = os.path.join(out_dir, file_name)
    if skip and os.path.exists(outfile_path):
        print(f"{file_name} exists, skip...")
        return
    command = (f'ffmpeg -ss 900 -t 901 -i "{os.path.join(in_dir, file_name)}"'
               f' -r 30 -strict experimental "{outfile_path}"')

    # print(command)
    subprocess.call(
        command,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    args = _parse_args()
    print(args)
    name_to_command_fn_dict = {
        "cut_videos": _cut_videos,
        "extract_frames": _extract_frames,
        "resize_videos": _resize_videos,
    }
    assert args.command_type in name_to_command_fn_dict
    assert os.path.exists(args.in_dir)
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    command_fn_args = [args.in_dir, args.out_dir, args.skip]
    if args.command_type == "resize_videos":
        command_fn_args.append(args.shortside)
    command_fn = partial(name_to_command_fn_dict[args.command_type],
                         *command_fn_args)
    in_list = os.listdir(args.in_dir)

    pool = Pool(args.num_processes)
    for _ in tqdm(
            pool.imap_unordered(command_fn, in_list), total=len(in_list)):
        pass
    pool.close()
    pool.join()
