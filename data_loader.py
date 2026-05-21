import torch
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
from corrupt import *

def get_loader(name_dataset, batch_size, train=True):

    # Computed with compute_mean_std.py
    mean_std = {
        'amazon': {
            'mean': [0.79235494, 0.7862071 , 0.78418255],
            'std':  [0.31496558, 0.3174693 , 0.3193569 ]
        },
        'dslr': {
            'mean': [0.47086468, 0.44865608, 0.40637794],
            'std':  [0.20395322, 0.19204104, 0.1996422 ]
        },
        'webcam': {
            'mean': [0.6119875 , 0.6187739 , 0.61730677],
            'std':  [0.25063968, 0.25554898, 0.25773206]
        }
    }

    data_transform = transforms.Compose([
            transforms.Resize((100, 100)), 
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean_std[name_dataset]['mean'],
            #                      std=mean_std[name_dataset]['std'])
        ])

    dataset = datasets.ImageFolder(root='./data/%s' % name_dataset,
                                   transform=data_transform)
    dataset_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size, shuffle=train,
                                                 num_workers=0)
    return dataset_loader

def get_loader_corrupt(name_dataset, batch_size, train=True):

    data_transform = transforms.Compose([
            transforms.Resize(100),
            corrupt({"gaussian_noise": 0.25, "shot_noise": 0,
                     "impulse_noise": 0,"speckle_noise": 0,
             "gaussian_blur": 0,"glass_blur": 0,
             "defocus_blur": 0,"zoom_blur": 0,
             "motion_blur": 0}),
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean_std[name_dataset]['mean'],
            #                      std=mean_std[name_dataset]['std'])
        ])

    dataset = datasets.ImageFolder(root='./data/%s' % name_dataset,
                                   transform=data_transform)
    dataset_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size, shuffle=train,
                                                 num_workers=0)
    return dataset_loader

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch
from torchvision import transforms

def get_loader_split(name_dataset, batch_size=8, seed=0):

    transform = transforms.Compose([
        transforms.Resize((100,100)),
        transforms.ToTensor()
    ])

    dataset = ImageFolder(root='./data/%s' % name_dataset, transform=transform)

    N = len(dataset)
    train_size = int(N * 0.7)
    val_size   = int(N * 0.15)
    test_size  = N - train_size - val_size

    g = torch.Generator().manual_seed(seed)

    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=g
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader