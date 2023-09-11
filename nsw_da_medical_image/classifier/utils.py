import json
import os
import pathlib
from typing import List, Dict, Tuple
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import nsw_da_medical_image.dataset_util as du

def video_from_dir(dir: str) -> du.Video:
    for vid in du.Video:
        if vid.directory == dir:
            return vid
    raise ValueError("Video directory not found.")

def get_weights(data_dir: str, json_file: str) -> torch.Tensor:
    kfold = json.load(open(json_file))
    videos = list(kfold["train"])
    class_dict: Dict[str, int] = {}

    for video in videos:
        info_video = pd.read_csv(f"{data_dir}embryo_dataset_annotations/{video}_phases.csv", header=None)
        classes = info_video[0].tolist()
        n_classes = (info_video[2] - info_video[1] + 1).tolist()
        count_classes_dict = dict(zip(classes, n_classes))

        for cl, n_cl in count_classes_dict.items():
            class_dict[cl] = class_dict.get(cl, 0) + n_cl

    weight_per_class: List[float] = []
    N = float(sum(class_dict.values()))

    for i in range(len(class_dict)):
        cl = du.Phase.from_idx(i).label
        weight_per_class.append(N / float(class_dict[cl]))

    return torch.tensor(weight_per_class)

def get_test_transforms(resize: Tuple[int, int] = (256, 256)) -> transforms.Compose:
    return transforms.Compose([transforms.Resize(resize), transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])

def get_dataloader(
    data_dir: str,
    mode: str,
    batch_size: int,
    json_file: str
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
        shuffle = True
    else:
        data_aug = get_test_transforms()
        shuffle = False

    data_set = du.NSWDataset(
        base_path,
        videos=[video_from_dir(file) for file in files],
        planes=[du.FocalPlane.F_0],
        transform=data_aug
    )

    return DataLoader(data_set, batch_size, shuffle=shuffle)


