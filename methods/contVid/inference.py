import os
import numpy as np
import argparse
import imageio
import torch

from einops import rearrange
from diffusers import DDIMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

import torchvision
from controlnet_aux.processor import Processor

from models.pipeline_controlvideo import ControlVideoPipeline
from models.util import save_videos_grid, read_video
from models.unet import UNet3DConditionModel
from models.controlnet import ControlNetModel3D
from models.RIFE.IFNet_HDv3 import IFNet

from pathlib import Path
from omegaconf import OmegaConf
from benchmark_loader import TextPoseBench_loader
from PIL import Image


# device = "cuda"
sd_path = "checkpoints/stable-diffusion-v1-5"
inter_path = "checkpoints/flownet.pkl"
controlnet_dict_version = {
    "v10":{
        "openpose": "checkpoints/sd-controlnet-openpose",
        "depth_midas": "checkpoints/sd-controlnet-depth",
        "canny": "checkpoints/sd-controlnet-canny",
    },
    "v11": {
    "softedge_pidinet": "checkpoints/control_v11p_sd15_softedge",
    "softedge_pidsafe": "checkpoints/control_v11p_sd15_softedge",
    "softedge_hed": "checkpoints/control_v11p_sd15_softedge",
    "softedge_hedsafe": "checkpoints/control_v11p_sd15_softedge",
    "scribble_hed": "checkpoints/control_v11p_sd15_scribble",
    "scribble_pidinet": "checkpoints/control_v11p_sd15_scribble",
    "lineart_anime": "checkpoints/control_v11p_sd15_lineart_anime",
    "lineart_coarse": "checkpoints/control_v11p_sd15_lineart",
    "lineart_realistic": "checkpoints/control_v11p_sd15_lineart",
    "depth_midas": "checkpoints/control_v11f1p_sd15_depth",
    "depth_leres": "checkpoints/control_v11f1p_sd15_depth",
    "depth_leres++": "checkpoints/control_v11f1p_sd15_depth",
    "depth_zoe": "checkpoints/control_v11f1p_sd15_depth",
    "canny": "checkpoints/control_v11p_sd15_canny",
    "openpose": "checkpoints/control_v11p_sd15_openpose",
    "openpose_face": "checkpoints/control_v11p_sd15_openpose",
    "openpose_faceonly": "checkpoints/control_v11p_sd15_openpose",
    "openpose_full": "checkpoints/control_v11p_sd15_openpose",
    "openpose_hand": "checkpoints/control_v11p_sd15_openpose",
    "normal_bae": "checkpoints/control_v11p_sd15_normalbae"
    }
}
# load processor from processor_id
# options are:
# ["canny", "depth_leres", "depth_leres++", "depth_midas", "depth_zoe", "lineart_anime",
#  "lineart_coarse", "lineart_realistic", "mediapipe_face", "mlsd", "normal_bae", "normal_midas",
#  "openpose", "openpose_face", "openpose_faceonly", "openpose_full", "openpose_hand",
#  "scribble_hed, "scribble_pidinet", "shuffle", "softedge_hed", "softedge_hedsafe",
#  "softedge_pidinet", "softedge_pidsafe"]

POS_PROMPT = "" # " ,best quality, extremely detailed, HD, ultra-realistic, 8K, HQ, masterpiece, trending on artstation, art, smooth"
NEG_PROMPT = None # "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer difits, cropped, worst quality, low quality, deformed body, bloated, ugly, unrealistic"



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Text description of target video")
    parser.add_argument("--video_path", type=str, required=True, help="Path to a source video")
    parser.add_argument("--output_path", type=str, default="./outputs", help="Directory of output")
    parser.add_argument("--condition", type=str, default="depth", help="Condition of structure sequence")
    parser.add_argument("--video_length", type=int, default=15, help="Length of synthesized video")
    parser.add_argument("--height", type=int, default=512, help="Height of synthesized video, and should be a multiple of 32")
    parser.add_argument("--width", type=int, default=512, help="Width of synthesized video, and should be a multiple of 32")
    parser.add_argument("--smoother_steps", nargs='+', default=[19, 20], type=int, help="Timesteps at which using interleaved-frame smoother")
    parser.add_argument("--is_long_video", action='store_true', help="Whether to use hierarchical sampler to produce long video")
    parser.add_argument("--seed", type=int, default=42, help="Random seed of generator")
    parser.add_argument("--version", type=str, default='v10', choices=["v10", "v11"], help="Version of ControlNet")
    parser.add_argument("--frame_rate", type=int, default=None, help="The frame rate of loading input video. Default rate is computed according to video length.")
    parser.add_argument("--temp_video_name", type=str, default=None, help="Default video name")
    
    args = parser.parse_args()
    return args


def get_config(config_path):
    """
    Loads the config file and updates it with the command line arguments.
    The model name is also updated. The config is then converted to a dictionary.
    """
    base_conf = OmegaConf.load(str(config_path))
    overrides = OmegaConf.from_cli()
    updated_conf = OmegaConf.merge(base_conf, overrides)
    return OmegaConf.to_container(updated_conf)

def set_seed(seed_value):
    """
    Set seed for reproducibility.
    """
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def main(loader, ndx, save_dir, seed):
    bench_data = loader[ndx]
    prompt = bench_data["prompt"]
    video_cond = bench_data["pose_seq"]
    codebook_ndx = bench_data["codebook_index"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_frames = video_cond.shape[0]

    # Height and width should be a multiple of 32
    # args.height = (args.height // 32) * 32    
    # args.width = (args.width // 32) * 32    

    # processor = Processor("openpose")
    controlnet_dict = controlnet_dict_version["v10"]
    
    tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").to(dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae").to(dtype=torch.float16)
    unet = UNet3DConditionModel.from_pretrained_2d(sd_path, subfolder="unet").to(dtype=torch.float16)
    controlnet = ControlNetModel3D.from_pretrained_2d(controlnet_dict["openpose"]).to(dtype=torch.float16)
    interpolater = IFNet(ckpt_path=inter_path).to(dtype=torch.float16)
    scheduler=DDIMScheduler.from_pretrained(sd_path, subfolder="scheduler")

    pipe = ControlVideoPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            controlnet=controlnet, interpolater=interpolater, scheduler=scheduler,
        )
    pipe.enable_vae_slicing()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to(device)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # Step 1. Read a video
    # video = read_video(video_path=args.video_path, video_length=num_frames, width=args.width, height=args.height, frame_rate=args.frame_rate)

    # Save source video
    # original_pixels = rearrange(video, "(b f) c h w -> b c f h w", b=1)
    # save_videos_grid(original_pixels, os.path.join(args.output_path, "source_video.mp4"), rescale=True)


    # Step 2. Parse a video to conditional frames
    # t2i_transform = torchvision.transforms.ToPILImage()
    # pil_annotation = []
    pil_annotation = [Image.fromarray(fr.astype(np.uint8)) for fr in video_cond]
    # for frame in video:
    #     pil_frame = t2i_transform(frame)
    #     pil_annotation.append(processor(pil_frame, to_pil=True))

    ############## DIRECTLY REPLACE YOUR VIDEO HERE `video_cond`
    # Save condition video
    # video_cond = [np.array(p).astype(np.uint8) for p in pil_annotation]
    # imageio.mimsave(os.path.join(args.output_path, f"{"openpose"}_condition.mp4"), video_cond, fps=8)

    # Reduce memory (optional)
    # del processor;

    # Step 3. inference

    if False:
        window_size = int(np.sqrt(video_cond.shape[0]))
        sample = pipe.generate_long_video(prompt, video_length=num_frames, frames=pil_annotation, 
                    num_inference_steps=50, smooth_steps=[19, 20], window_size=window_size,
                    generator=generator, guidance_scale=12.5, negative_prompt=NEG_PROMPT,
                    width=512, height=512
                ).videos
    else:
        sample = pipe(prompt, video_length=num_frames, frames=pil_annotation, 
                    num_inference_steps=50, smooth_steps=[19, 20],
                    generator=generator, guidance_scale=12.5, negative_prompt=NEG_PROMPT,
                    width=512, height=512
                ).videos
    sample = sample.squeeze(0).permute(1,2,3,0)
    result = np.stack([(fr*255).numpy().astype(np.uint8) for fr in sample]).astype(np.uint8)
    filename = "{:05d}.npy".format(codebook_ndx)
    np.save(save_dir/filename, result)


if __name__ == "__main__":
    config = get_config("bench_runner.yaml")
    loader = TextPoseBench_loader(config)
    seed = config["seed"]
    save_path = Path(config["save_path"])
    save_dir = save_path/("".join(config["codebook_name"].split(".")[:-1]))
    os.makedirs(save_dir, exist_ok=True)

    for ndx in range(len(loader)):
        set_seed(seed)
        main(loader, ndx, save_dir, seed)
        torch.cuda.empty_cache()
