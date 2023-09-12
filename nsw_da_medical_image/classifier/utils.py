import json
import os
import pathlib
from typing import List, Dict, Tuple
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import nsw_da_medical_image.dataset_util as du
import numpy as np

def video_from_dir(dir: str) -> du.Video:
    for vid in du.Video:
        if vid.directory == dir:
            return vid
    raise ValueError("Video directory not found.")


def get_weights_per_image(base_path,videos,data_aug):
        
    data_set = du.NSWDataset(
        base_path,
        videos=[video_from_dir(video) for video in videos],
        planes=[du.FocalPlane.F_0],
        transform=data_aug)
          
    phases =[phase.idx() for phase in data_set.all_phases()]
    classes, count_classes = np.unique(phases, return_counts=True)

    weight_per_class = {}                                    
    N = count_classes.sum()

    for i in range(len(classes)):
        weight_per_class[classes[i]] = N/float(count_classes[i])
    
    weights_per_image = [weight_per_class[phase] for phase in phases]

    return torch.utils.data.sampler.WeightedRandomSampler(weights_per_image, len(weights_per_image)), data_set

def get_test_transforms(resize: Tuple[int, int] = (256, 256)) -> transforms.Compose:
    return transforms.Compose([transforms.Resize(resize), transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])

def get_dataloader(
    data_dir: str,
    mode: str,
    batch_size: int,
    json_file: str,
    select_median_only: bool = False,
) -> DataLoader:
    kfold = json.load(open(json_file))
    files = list(kfold[mode])
    base_path = pathlib.Path(data_dir)

    if mode == "train":
        data_aug = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.RandomRotation(180),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])
        sampler, data_set = get_weights_per_image(base_path,files,data_aug)

    else:
        data_aug = get_test_transforms()
        sampler = None
        data_set = du.NSWDataset(
            base_path,
            videos=[video_from_dir(file) for file in files],
            planes=[du.FocalPlane.F_0],
            transform=data_aug,
            select_median_only=select_median_only,
        )
    

    return DataLoader(data_set, batch_size, sampler = sampler)


