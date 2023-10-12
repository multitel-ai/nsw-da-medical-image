import torch
from pathlib import Path

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel


def generate(
    model_path: Path,
    prompt: str,
    num_images: int,
    image_destination: Path,
):
    "generate some images in image_destination using prompt"
    cuda_available = torch.cuda.is_available()

    # load the models from the config, no need for accelerate here
    torch_dtype = torch.float16 if cuda_available else torch.float32
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=torch_dtype)
    pipe = DiffusionPipeline.from_pretrained(model_path, unet=unet,torch_dtype=torch_dtype)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda" if cuda_available else "cpu")

    for idx in range(num_images):
        # batch-size is 1 so we only take the first image
        image = pipe(prompt).images[0]
        image.save(image_destination / f"img{idx}.jpg")
