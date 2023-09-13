import random
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torchvision import transforms, models
from datetime import datetime
from pathlib import Path
import wandb
from .model import build_model
from .utils import get_dataloader, get_weights
import typing

# Ensuring Reproducibility
def set_seed():
    torch.random.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    if torch.cuda.is_available():
        torch.multiprocessing.set_sharing_strategy('file_system')
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.benchmark = False

def get_device():
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name())
        return torch.device('cuda')
    else:
        print('GPU not available')
        # Uncomment the below line if you want to raise an error when GPU is not available.
        # raise Error("Cannot train the network on CPU. Make sure you have a GPU and that it is available.")
        return torch.device('cpu')


def run_train(num_epochs: int,
              lr: float,
              batch_size: int,
              weights: str,
              data_dir: str,
              json_file: str,
              architecture: str = typing.Literal['resnet50', 'densenet121'],
              wandb_project_name: str = None,
              wandb_run_name: str = None,
              save_dir: str = '/App/models',
              freeze: bool = False):
    set_seed()
    device = get_device()
    model = build_model(net=architecture, path=weights)
    model = model.to(device)
    loss = nn.CrossEntropyLoss(weight=get_weights(data_dir,json_file).to(device))
    
    if freeze:
        if "resnet" in architecture:
            last_layer_name = "fc" 
            params = model.fc.parameters()
        elif "densenet" in architecture:
            last_layer_name = "classifier"
            params = model.classifier.parameters()
        else:
            raise Exception(f"Architecture not recognized: '{architecture}'")
        for name, param in model.named_parameters():
            if last_layer_name not in name:
                param.requires_grad = False
    else:
        params = model.parameters()
    
    optimizer = Adam(params, lr=lr)
    
    now = datetime.now()
    formatted_string = now.strftime("%d-%m_%Hh%M")
    save_dir = Path(save_dir) / (formatted_string + "_" + wandb_run_name)
    save_dir.mkdir(exist_ok=True)
    (save_dir/"best").mkdir(exist_ok=True)
    (save_dir/"checkpoint").mkdir(exist_ok=True)
    
    tr_loader = get_dataloader(data_dir, "train", batch_size, json_file)
    val_loader = get_dataloader(data_dir, "val", batch_size, json_file)
    
    train_dataset_length = len(tr_loader)
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs*train_dataset_length)
    
    wandb.login()
    wandb.init(project=wandb_project_name, mode="online", name=wandb_run_name, 
            entity='trail23-medical-image-diffusion')

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        step = 0
        print("-" * 10)
        print(f"epoch {epoch + 1}/{num_epochs}")
				
        model.train()
        
        for tidx, (images, phase, plane, video, frame) in enumerate(tr_loader):
            step += 1 
            images = images.to(device)
            labels = phase.to(device)

            optimizer.zero_grad(set_to_none=True)     

            labels_pred = model(images)

            loss_pred = loss(labels_pred, labels).mean()
            loss_pred.backward()
            
            epoch_loss += loss_pred.item()
            optimizer.step()
            
            lr_scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            if step % 100 == 0:
                print(f"{step}/{len(tr_loader)}, train_loss: {loss_pred.item():.4f} // Current lr: {current_lr}")

            wandb.log(
    		      {'Training Loss/Total Loss': loss_pred.item(), 'Learning rate': current_lr}, 
    		      step=(epoch*train_dataset_length)+tidx)
        
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'wandb_run_id': wandb.run.id,
            'scheduler': lr_scheduler.state_dict()
        }, save_dir/"checkpoint"/ (wandb_run_name + "checkpoint.pth"))
        
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")


        model.eval()
        best_acc = 0.0
        with torch.no_grad():
            print("Validation...")
            acc = 0.0
            epoch_loss_val = 0.0
            for vidx, (images, phase, _, _, _) in enumerate(val_loader):
                images = images.to(device)
                labels = phase.to(device)
    
                labels_pred = model(images)
                acc += (labels_pred.argmax(dim=1) == labels).float().mean().item()
                epoch_loss_val += loss(labels_pred, labels).mean().item()

            acc /= (vidx+1)
            epoch_loss_val /= len(val_loader)
            
            print(f"Accuracy after this epoch: {round(acc,4)}")
            
            if acc > best_acc:
                torch.save(model.state_dict(), save_dir / "best" / ( wandb_run_name + "_best_acc.pth"))
    
            wandb.log(
                {'Validation/Total Loss': epoch_loss_val,
                 'Validation/Accuracy': acc},
                step=(epoch*train_dataset_length)+tidx)



