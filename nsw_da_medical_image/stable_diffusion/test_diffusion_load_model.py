import torch
import os 
import argparse, sys
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel
from transformers import CLIPTextModel

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))



def main( n_iter, version, prompt, output_dir):

    accelerator = Accelerator()

    output_folder = os.path.join(output_dir,f"{version}")

    os.makedirs(output_folder, exist_ok=True)

    run_id = 0

    while os.path.isdir(os.path.join(output_folder, f"run_{run_id}")):
        run_id += 1 

    output_folder_run = os.path.join(output_folder, f"run_{run_id}" )
    os.mkdir(output_folder_run) 

    model_id = "stabilityai/stable-diffusion-2-1-base"
    
    unet = UNet2DConditionModel.from_pretrained("/App/models/stable_diffusion/longer-trained-phase-tPB2", subfolder="unet")
    #text_encoder = CLIPTextModel.from_pretrained("/App/models/stable_diffusion/longer-trained-phase-tPB2", subfolder='text_encoder')

    pipe = DiffusionPipeline.from_pretrained(model_id, unet=accelerator.unwrap_model(unet))
    pipe.to("cuda")
    
    for i in range(n_iter):
        image = pipe(prompt).images[0]
        image.save(os.path.join(output_folder_run,f"embryo{i}.png"))

if __name__ == '__main__':

    parser=argparse.ArgumentParser()

    parser.add_argument("--v", help="Model version", )
    parser.add_argument("--p", help="Custom Prompt",)
    parser.add_argument('--o', help='path to where synthetic images would be saved')

    
    args=parser.parse_args()

    m_version = 'checkpoint_done_2'
    n = 15
    p = 'a grayscale microscopic image of human embryo at phase tPB2'
    o = '/App/data/synthetic-images'

    main(n_iter=n, version=m_version, prompt=p, output_dir=o)
