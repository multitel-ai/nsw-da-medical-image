# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 14:49:47 2023

@author: Laura
"""

import pathlib
import dataset_util as du
from torch.utils.data import DataLoader
import json
from torchvision.transforms import Resize, ToTensor, Compose, RandomHorizontalFlip, RandomVerticalFlip,RandomRotation
from torchvision import transforms
import pandas as pd
import os
import torch

def video_from_dir(dir: str) -> du.Video:
    for vid in du.Video:
        if vid.directory == dir:
            return vid
    raise ValueError("")
    

    
def make_weights_for_balanced_classes(dir):
    class_dict = {}
    video_classes = {}
    videos = os.listdir(f"{dir}embryo_dataset/")
    for video in videos:
        info_video = pd.read_csv(f"{dir}embryo_dataset_annotations/{video}_phases.csv",header=None)
        classes = info_video[0].tolist()
        n_classes = (info_video[2]-info_video[1]+1).tolist()
        count_classes_dict = dict(zip(classes,n_classes))
        video_classes[video] = count_classes_dict
        
        for cl,n_cl in count_classes_dict.items():
            if cl not in class_dict.keys():
                class_dict[cl] = n_cl
            else:
                class_dict[cl]+=n_cl
                
    weight_per_class = {}                                    
    N = float(sum(class_dict.values())) 
    
    for cl,n_cl in class_dict.items():
        weight_per_class[cl] = N/float(n_cl)
    
    weights = {}
    for video, count in video_classes.items():
        for cl,n_cl in count.items():
            weights[video] = n_cl*weight_per_class[cl]
        
    return weights
   
def get_dataloader(data_dir:str,
                   mode:str,
                   batch_size:int,
                   json_file:str):
    base_path = pathlib.Path(data_dir)

    kfold = json.load(open(json_file))
    files = list(kfold[mode].keys())

    
    if mode=="train_set":
        data_aug = transforms.Compose([Resize((256, 256)), RandomHorizontalFlip(),
                                         RandomVerticalFlip(),RandomRotation(90), 
                                         RandomRotation(180), ToTensor()])
        
        weights=list(kfold[mode].values())
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    else:
        data_aug = transforms.Compose([Resize((256, 256)), ToTensor()])
        sampler = None
        
    data_set = du.NSWDataset(
        base_path,
        videos=[video_from_dir(file) for file in files],
        planes=[du.FocalPlane.F_0],
        transform=Compose(data_aug))
    
    
    dataloader = DataLoader(data_set, batch_size, sampler = sampler)
    return dataloader


