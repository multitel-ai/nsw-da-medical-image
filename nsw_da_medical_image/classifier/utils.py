# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 14:49:47 2023

@author: Laura
"""

import pathlib
import nsw_da_medical_image.dataset_util as du
from torch.utils.data import DataLoader
import json
from torchvision.transforms import Resize, ToTensor, Compose, RandomHorizontalFlip, RandomVerticalFlip,RandomRotation, Grayscale
from torchvision import transforms
import pandas as pd
import os
import torch

def video_from_dir(dir: str) -> du.Video:
    for vid in du.Video:
        if vid.directory == dir:
            return vid
    raise ValueError("")
    

    
def get_weights(data_dir:str, json_file:str):
    
    kfold = json.load(open(json_file))
    videos = list(kfold["train"])
    class_dict = {}

    for video in videos:
        info_video = pd.read_csv(f"{data_dir}embryo_dataset_annotations/{video}_phases.csv",header=None)
        classes = info_video[0].tolist()
        n_classes = (info_video[2]-info_video[1]+1).tolist()
        count_classes_dict = dict(zip(classes,n_classes))
        
           
        for cl,n_cl in count_classes_dict.items():
            if cl not in class_dict.keys():
                class_dict[cl] = n_cl
            else:
                class_dict[cl]+=n_cl
                    
    weight_per_class = []                                    
    N = float(sum(class_dict.values())) 

    for i in range(len(class_dict.items())):
        cl = du.Phase.from_idx(i).label
        weight_per_class.append( N/float(class_dict[cl]))
            
        

    return torch.tensor(weight_per_class)

def get_test_transforms(resize: tuple = (256,256)) -> transforms.Compose:
    return transforms.Compose([Resize(resize), Grayscale(num_output_channels=3), ToTensor()])
   
def get_dataloader(data_dir:str,
                   mode:str,
                   batch_size:int,
                   json_file:str):

    kfold = json.load(open(json_file))
    files = list(kfold[mode])
    base_path = pathlib.Path(data_dir)
    
    if mode=="train":
        data_aug = transforms.Compose([Resize((256, 256)), RandomHorizontalFlip(),
                                         RandomVerticalFlip(),RandomRotation(90), 
                                         RandomRotation(180), Grayscale(num_output_channels=3),
                                         ToTensor()])
        shuffle = True
        
    else:
        data_aug = get_test_transforms() 
        
        shuffle= False
        
    data_set = du.NSWDataset(
        base_path,
        videos=[video_from_dir(file) for file in files],
        planes=[du.FocalPlane.F_0],
        transform=data_aug)
    
    
    dataloader = DataLoader(data_set, batch_size, shuffle = shuffle)
    return dataloader


