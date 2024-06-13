from benchmark_loader import get_config
from multiprocessing import Process, current_process
from moviepy.editor import ImageClip, concatenate_videoclips
from pathlib import Path

import gc
import numpy as np
import os
import pandas as pd


def create_grid_frames(ncols, frames_list):
    cols = list()
    rows = list()
    for start in range(0, len(frames_list), ncols):
        cols.append(frames_list[start:start+ncols])
    for col in cols:
        rows.append(np.concatenate(col, axis=2))
    grid_frames = np.concatenate(rows, axis=1)
    return grid_frames


def create_mp4(grid_frames, save_path, outfilename, fps=24):
    save_path.mkdir(parents=True, exist_ok=True)
    imgClips = [ImageClip(frame).set_duration(1/fps) for frame in grid_frames]
    concat_clip = concatenate_videoclips(imgClips, method="compose")
    concat_clip.write_videofile(str(save_path/(outfilename+".mp4")), fps=fps, verbose=False, logger=None)
    print("MP4 {}/{} saved".format(save_path, outfilename))


def get_frames(filepath, filename):
    frames = np.load(filepath/(filename+".npy"))
    return frames


def get_codebook(codebook_path, codebook_name):
    df = pd.read_csv(str(codebook_path/(codebook_name+".csv")))
    return df


def runner(base_path, save_path, vids, codebook):
    for vid_fname in vids:
        vid_fname = "".join(vid_fname.split(".")[:-1])
        prompt = codebook.iloc[int(vid_fname)]["prompt"]
        fps = codebook.iloc[int(vid_fname)]["sampling_fps"]
        save_name = "{}_{}".format(vid_fname, "_".join([fn.strip(".").strip(",") for fn in prompt.split(" ")]))
        frames = get_frames(base_path, vid_fname)
        create_mp4(frames, save_path, save_name, fps=fps)
        gc.collect()


if __name__ == "__main__":
    cfg = get_config("bench_runner.yaml")

    codebook_name = "".join(cfg["codebook_name"].split(".")[:-1])
    data_path = Path(cfg["data_path"])
    inference_path = Path(cfg["save_path"])/codebook_name
    save_path = Path(str(inference_path) + "__videos")

    vid_list = os.listdir(inference_path)
    vid_list.sort()
    codebook = get_codebook(data_path/cfg["codebook_dir"], codebook_name)

    num_parallel_runners = 15
    splits = list(np.array_split(vid_list, num_parallel_runners))
    processes = list()
    for ndx, split in enumerate(splits[:-1]):
        p = Process(target = runner, args=(inference_path, save_path, split, codebook), name="proc_{}".format(ndx))
        p.start()
        processes.append(p)
        print("Started processes_{}".format(ndx))
    runner(inference_path, save_path, splits[-1], codebook)
    for p in processes:
        p.join()
    print("Finished converting the codebook inferences to mp4")
    breakpoint()
