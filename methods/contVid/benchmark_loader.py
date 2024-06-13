from omegaconf import OmegaConf
from pathlib import Path
from torch.utils.data import Dataset

import math
import numpy as np
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


class TextPoseBench_loader(Dataset):
    def __init__(self, config):
        self.config = config
        self.data_path = Path(self.config["data_path"])
        self.vid_dir = self.data_path/self.config["pose_vid_dir"]
        self.codebook = self.get_codebook(self.config["codebook_dir"], self.config["codebook_name"])
        self.ndxs = self.get_dataloader_ndxs(self.config["num_chunks"], self.config["chunk_index"])
        print("Loader ready | Start index: {} | End index: {} | Total rows: {}".format(self.ndxs[0], self.ndxs[-1], len(self.ndxs)))

    def get_codebook(self, codebook_dir, codebook_name):
        codebook = pd.read_csv(str(self.data_path/codebook_dir/codebook_name))
        return codebook

    def get_dataloader_ndxs(self, num_chunks, chunk_index):
        assert num_chunks == -1 or num_chunks > chunk_index, "Chunk index cannot be greater or equal to the number of chunks"
        assert chunk_index >= 0, "Chunk index cannot be negative"
        ndxs = np.array(self.codebook.index.to_list())
        total_rows = ndxs.shape[0]
        if num_chunks != -1:
            chunk_size = math.ceil(total_rows/num_chunks)
            start = chunk_index*chunk_size
            end = start + chunk_size
            ndxs = ndxs[start:end]
        return ndxs

    def __getitem__(self, ndx):
        codebook_ndx = self.ndxs[ndx]
        row = self.codebook.iloc[codebook_ndx]
        pose_seq_name = row["filename"]
        prompt = row["prompt"]
        pose_seq = np.load(self.data_path/self.vid_dir/pose_seq_name)
        data = {
            "prompt": prompt,
            "pose_seq": pose_seq,
            "codebook_index": codebook_ndx,
        }
        return data

    def __len__(self):
        return len(self.ndxs)


if __name__ == "__main__":
    cfg = get_config("configs/bench_runner.yaml")
    loader = TextPoseBench_loader(cfg)
