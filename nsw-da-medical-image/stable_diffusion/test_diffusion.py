import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here insteasd
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "Medical image with optical micropscope of a human embryo at development stage t2"
for i in range(3):
    image = pipe(prompt).images[0]
    image.save(f"/home/ucl/ingi/echatzop/Nantes/nsw-da-medical-image/nsw-da-medical-image/stable_diffusion/embryo{i}.png")