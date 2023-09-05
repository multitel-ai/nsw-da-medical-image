import torch
import os 

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

#model_id = "stabilityai/stable-diffusion-2-1"
model_id = os.path.join(__location__,'models/v1.0')

unet = UNet2DConditionModel

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here insteasd
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "Medical image with optical microscope of a human embryo at development stage t2"
for i in range(3):
    image = pipe(prompt).images[0]
    image.save(os.path.join(__location__,'generated_images',f"embryo{i}.png"))

