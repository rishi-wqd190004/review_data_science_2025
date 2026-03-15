#%%
## Apply data augmentation on CIFAR-10 dataset

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader

from helper_visualize import unnormalize, show_augmented_vs_org

# transformations
train_transformation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

val_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

raw_dataset = torchvision.datasets.CIFAR10(root='/home/rishi/review_data_science_2025/computer_vision/src/ch_03/data', train=True)

train_dataset = torchvision.datasets.CIFAR10(root='/home/rishi/review_data_science_2025/computer_vision/src/ch_03/data', train=True, transform=train_transformation, download=True)
val_dataset = torchvision.datasets.CIFAR10(root='/home/rishi/review_data_science_2025/computer_vision/src/ch_03/data', train=False, transform=val_transformation, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

classes = train_dataset.classes

# view the augmentation
show_augmented_vs_org(transform=train_transformation, dataset=raw_dataset)
# %%
