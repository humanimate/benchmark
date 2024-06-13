from pathlib import Path
from diffusers import StableDiffusionPipeline, ControlNetModel

import torch


if __name__ == "__main__":
    base_path = Path("./")
    ckpt_dir = "checkpoints"
    hf_cachedir = None

    sdv15_outdir = "SD_v1_5_freshDown"
    contNet_v0_outdir = "controlNet_v0_openPose"

    sdv15_savepath = base_path/ckpt_dir/sdv15_outdir
    contNet_v0_savepath = base_path/ckpt_dir/contNet_v0_outdir

    print("Initializing the SD v1.5 pipe, and ControlNet")
    sdv15_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", cache_dir="/sensei-fs/tenants/Sensei-AdobeResearchTeam/share-dsrivastava/hf_cache")
    controlnet_v0 = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", cache_dir="/sensei-fs/tenants/Sensei-AdobeResearchTeam/share-dsrivastava/hf_cache")

    print("Saving pipe components")
    sdv15_pipe.save_pretrained(str(sdv15_savepath))

    print("Saving ControlNet_v0-openpose")
    controlnet_v0.save_pretrained(str(contNet_v0_savepath))

    print("Components saved in dir: {}".format(base_path/ckpt_dir))
