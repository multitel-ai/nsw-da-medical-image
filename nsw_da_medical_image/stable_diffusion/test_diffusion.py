import torch
import os 
import argparse, sys
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))



def main(model_id, n_iter, version, prompt, output_dir):

    output_folder = os.path.join(output_dir,f"{version}")

    os.makedirs(output_folder, exist_ok=True)

    run_id = 0

    while os.path.isdir(os.path.join(output_folder, f"run_{run_id}")):
        run_id += 1 

    output_folder_run = os.path.join(output_folder, f"run_{run_id}" )
    os.mkdir(output_folder_run) 
    
    logging_dir = Path(model_id, 'logs')

    accelerator_project_config = ProjectConfiguration(project_dir=model_id, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="no",
        log_with='wandb',
        project_config=accelerator_project_config
    )
    torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch_dtype)
    pipe = DiffusionPipeline.from_pretrained(model_id, unet=accelerator.unwrap_model(unet),torch_dtype=torch_dtype)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

   
    for i in range(n_iter):
        image = pipe(prompt).images[0]
        image.save(os.path.join(output_folder_run,f"embryo{i}.png"))

if __name__ == '__main__':

    parser=argparse.ArgumentParser()

    parser.add_argument("--m", help="Model path",)
    parser.add_argument("--n", help="Num iterrations",)
    parser.add_argument("--v", help="Model version", )
    parser.add_argument("--p", help="Custom Prompt",)
    parser.add_argument('--o', help='path to where synthetic images would be saved')

    
    args=parser.parse_args()

    model = args.m
    m_version = args.v
    n = int(args.n)
    p = args.p
    o = args.o

    main(model_id=model, n_iter=n, version=m_version, prompt=p, output_dir=o)
