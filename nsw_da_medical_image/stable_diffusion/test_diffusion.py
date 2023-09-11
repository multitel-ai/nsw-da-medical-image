import torch
import os 
import argparse, sys

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))



def main(model_id, n_iter, version, prompt):

    output_folder = os.path.join(__location__,'all-synthetic-runs',f"{version}")

    os.makedirs(output_folder, exist_ok=True)

    run_id = 0

    while os.path.isdir(os.path.join(output_folder, f"run_{run_id}")):
        run_id += 1 

    output_folder_run = os.path.join(output_folder, f"run_{run_id}" )
    os.mkdir(output_folder_run)


    #model_id = "stabilityai/stable-diffusion-2-1"
    

    

    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here insteasd
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

   
    for i in range(n_iter):
        image = pipe(prompt).images[0]
        image.save(os.path.join(output_folder_run,f"embryo{i}.png"))

if __name__ == '__main__':

    parser=argparse.ArgumentParser()

    parser.add_argument("--m", help="Model path", default = os.path.join(__location__,'models/v1.1'))
    parser.add_argument("--n", help="Num iterrations", default = 3)
    parser.add_argument("--v", help="Model version", default = "v1_1")
    parser.add_argument("--p", help="Custom Prompt", default = "Medical image with optical microscope of a human embryo at development stage t2")

    
    args=parser.parse_args()

    model = args.m
    m_version = args.v
    n = int(args.n)
    p = args.p

    main(model_id=model, n_iter=n, version=m_version, prompt=p)