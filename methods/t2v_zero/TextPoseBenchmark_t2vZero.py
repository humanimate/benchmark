from benchmark_loader import TextPoseBench_loader
from model import Model
from omegaconf import OmegaConf
from pathlib import Path

import numpy as np
import torch

from PIL import Image


class t2vZero_benchmark(object):
    def __init__(self, config):
        self.config = config
        self.seed = config["seed"]
        self.guidance_scale = config["guidance_scale"]
        self.loader = TextPoseBench_loader(self.config)
        self.device = torch.device("cuda:{}".format(config["gpu_id"]) if torch.cuda.is_available() else "cpu")
        self.model = Model(device = self.device, dtype = torch.float16)
        self.save_path = Path(config["save_path"])
        self.save_dir = self.save_path/("".join(config["codebook_name"].split(".")[:-1]))
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def draw_inference(self, prompt, pose_vid):
        result = self.model.process_controlnet_pose(
                                video_path=pose_vid,
                                prompt=prompt, save_path="./",
                                num_inference_steps=50,
                                guidance_scale=self.guidance_scale,
                                seed=self.seed,
                        )
        result = np.stack([(fr*255).astype(np.uint8) for fr in result]).astype(np.uint8)
        return result

    def run_benchmark(self):
        for ndx in range(len(self.loader)):
            bench_data = self.loader[ndx]
            prompt = bench_data["prompt"]
            pose_vid = torch.Tensor(bench_data["pose_seq"]).permute(0,3,1,2).to(self.device)
            codebook_ndx = bench_data["codebook_index"]
            res = self.draw_inference(prompt, pose_vid)
            self.save_inferences(res, codebook_ndx)

    def save_inferences(self, inference, codebook_ndx):
        filename = "{:05d}.npy".format(codebook_ndx)
        np.save(self.save_dir/filename, inference)

if __name__ == "__main__":
    cfg = OmegaConf.load("bench_runner.yaml")
    obj = t2vZero_benchmark(cfg)
    obj.run_benchmark()
