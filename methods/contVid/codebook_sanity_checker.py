from omegaconf import OmegaConf
from pathlib import Path

import os
import pandas as pd


def get_config(config_path):
    """
    Loads the config file and updates it with the command line arguments.
    The model name is also updated. The config is then converted to a dictionary.
    """
    base_conf = OmegaConf.load(str(config_path))
    overrides = OmegaConf.from_cli()
    updated_conf = OmegaConf.merge(base_conf, overrides)
    return OmegaConf.to_container(updated_conf)


def check_video_list(vid_path, vid_titles):
    cached_vid_list = os.listdir(vid_path)
    cached_vid_list.sort()
    flag = False
    for vid_title in vid_titles:
        if vid_title not in cached_vid_list:
            print("{}/{} not found".format(vid_path, vid_title))
            flag = True
            break
    print("All required videos are here!" if not flag else "Issue found in during video list check")


def get_codebook(codebook_path, codebook_name):
    df = pd.read_csv(str(codebook_path/codebook_name))
    return df


if __name__ == "__main__":
    cfg = get_config("bench_runner.yaml")

    data_path = Path(cfg["data_path"])
    vid_path = data_path/cfg["pose_vid_dir"]
    codebook_path = data_path/cfg["codebook_dir"]

    codebook_name = cfg["codebook_name"]
    codebook = get_codebook(codebook_path, codebook_name)

    vid_titles = codebook["filename"].to_list()
    check_video_list(vid_path, vid_titles)
