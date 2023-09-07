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
    

    
def get_weights_dataset(files,dir,data_aug, mode):
    base_path = pathlib.Path(dir)
    if mode == "train":
        class_dict = {}
        weights_per_image = []
        for video in files:
            info_video = pd.read_csv(f"{dir}embryo_dataset_annotations/{video}_phases.csv",header=None)
            classes = info_video[0].tolist()
            n_classes = (info_video[2]-info_video[1]+1).tolist()
            count_classes_dict = dict(zip(classes,n_classes))
            
            
               
            for cl,n_cl in count_classes_dict.items():
                if cl not in class_dict.keys():
                    class_dict[cl] = n_cl
                else:
                    class_dict[cl]+=n_cl
                        
        weight_per_class = {}                                    
        N = float(sum(class_dict.values())) 
    
        for cl,n_cl in class_dict.items():
            weight_per_class[cl] = N/float(n_cl)
            
        data_set = du.NSWDataset(
            base_path,
            videos=[video_from_dir(file) for file in files],
            planes=[du.FocalPlane.F_0],
            transform=data_aug)
              
        #for img,phase, plane,video,frame_number in data_set:
        #    weights_per_image.append(weight_per_class[du.Phase.from_idx(phase).label])


        return data_set,weights_per_image
    else:
        data_set = du.NSWDataset(
            base_path,
            videos=[video_from_dir(file) for file in files],
            planes=[du.FocalPlane.F_0],
            transform=data_aug)
        return data_set
        
   
def get_dataloader(data_dir:str,
                   mode:str,
                   batch_size:int,
                   json_file:str):

    kfold = json.load(open(json_file))
    files = list(kfold[mode])

    
    if mode=="train":
        data_aug = transforms.Compose([Resize((256, 256)), RandomHorizontalFlip(),
                                         RandomVerticalFlip(),RandomRotation(90), 
                                         RandomRotation(180), Grayscale(num_output_channels=3),
                                         ToTensor()])
        
        data_set, weights = get_weights_dataset(files,data_dir,data_aug, mode)
        sampler = None#torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    else:
        data_aug = transforms.Compose([Resize((256, 256)), Grayscale(num_output_channels=3), ToTensor()])
        data_set = get_weights_dataset(files,data_dir,data_aug, mode)
        sampler = None
    
    
    
    dataloader = DataLoader(data_set, batch_size, sampler = sampler)
    return dataloader


