# -*- coding: utf-8 -*-


import torch
import numpy
from nsw_da_medical_image.dataset_util.dataset import NSWDataset
from torchvision import transforms,models 
import pathlib
import nsw_da_medical_image.dataset_util as du
import torch
from torch import nn
from torch.utils.data import dataloader
from torch.optim import Adam, Optimizer
import numpy as np
import random
import json
import argparse
import os
from nsw_da_medical_image.classifier.model import build_model
import wandb
from nsw_da_medical_image.classifier.utils import get_dataloader

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0',
                    help='GPU device', )
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size')
parser.add_argument('--num_epochs', type=int, default=50,
                    help='Max number of epochs')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate')
parser.add_argument('--data_dir', type=str,
                    help='Data path')
parser.add_argument('--json_file', type=str, default="data_split.json",
                    help='Json file with train/val split')
parser.add_argument('--pretrained_weights', type=str, default="pretrained",
                    help='Set pretrained weights')
parser.add_argument('--wandb_project', type=str, default='classifier', 
                    help='wandb project name')
parser.add_argument('--name', type=str, default='Give me a name !', 
                    help='wandb run name')



args = parser.parse_args()
args_pool = {
            'n_class':len(du.Phase),
            'channels':1,
            'input_size': 256,
            'train_batch_size': args.batch_size,
            'val_batch_size': 32, 
        }


    
############################# For reproducibility #############################################
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
torch.random.manual_seed(0)
np.random.seed(0)
random.seed(0)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name())
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.benchmark = False
else:
    print('GPU not available')
###############################################################################################


def run_train(num_epochs: int,
              lr: float,
              weights: str,
              dev: torch.device,
              data_dir: str,
              wandb_project_name: str = None,
              wandb_run_name: str = None):
    dev = "cuda:"+dev
    mdl = build_model(weights)
    mdl = mdl.to(dev)
    loss = nn.CrossEntropyLoss()
    optim = Adam(mdl.parameters(), lr=lr)
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=num_epochs)
    tr_loader = get_dataloader(data_dir,"train",args_pool['train_batch_size'],args.json_file)
    val_loader = get_dataloader(data_dir,"valid",args_pool['valid_batch_size'])
    
    wandb.login()
    wandb.init(project=wandb_project_name, mode="online", name=wandb_run_name, 
            entity='trail23-medical-image-diffusion')
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        step = 0
        print("-" * 10)
        print(f"epoch {epoch + 1}/{num_epochs}")
				
        mdl.train()
        
        for idx, (img, phase, plane, video) in enumerate(tr_loader):
            step += 1 
            inp = img.to(dev)
            lbl= phase.to(dev)

            optim.zero_grad(set_to_none=True)     

            pred = mdl(inp)

            loss_pred = loss(pred, lbl).mean()
            loss_pred.backward()
            
            epoch_loss += loss_pred.item()
            optim.step()

            if step % 100 == 0:
                print(f"{step}/{len(tr_loader)}, train_loss: {loss_pred.item():.4f}")

        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        lr_scheduler.step()
        current_lr = optim.param_groups[0]["lr"]

        wandb.log(
    		{'Training Loss/Total Loss': epoch_loss, 'Learning rate': current_lr}, 
    		step=epoch) 


        mdl.eval()
        with torch.no_grad():
            acc = 0.0
            epoch_loss_val = 0.0
            for _, (img, phase, _, _) in enumerate(val_loader):
                inp = img.to(dev)
                lbl = phase.to(dev)
    
                pred = mdl(inp)
                acc += (pred.argmax(dim=1) == lbl).float().mean().item()
                epoch_loss_val += loss(pred, lbl).mean().item()
            
            acc /= len(val_loader)
            
            epoch_loss_val /= len(val_loader)
    
        wandb.log(
            {'Validation/Total Loss': epoch_loss_val,
             'Validation/Accuracy': acc},
            step=epoch)
                
if __name__ == "__main__":

    run_train(num_epochs=args.num_epochs,lr=args.lr, weights=args.pretrained_weights, 
              dev=args.device, data_dir=args.data_dir,wandb_project_name=args.wandb_project, 
              wandb_run_name=args.name)



