import torch
import torchvision
import kornia.augmentation as K
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

from helper import visualize_smpl

def get_device():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu_name} (count: {torch.cuda.device_count()})")
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon)")
        return torch.device('mps')
    else:
        print("Using CPU")
        return torch.device('cpu')

# base transformation to get data into CHW, 0-1 range 
base_transformation = T.ToTensor()

train_transformation = K.AugmentationSequential(
    # 1. Geometric
    K.RandomHorizontalFlip(p=0.5),
    K.RandomCrop((32,32), padding=4, p=1.0),
    K.RandomRotation(degrees=15.0, p=0.5),
    # 2. Photometric
    K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    # 3. Noise/Regularization
    K.RandomGaussianNoise(mean=0.0, std=0.01, p=0.2),
    # 4. Normalization
    K.Normalize(mean=torch.tensor([0.4914, 0.4822, 0.4465]), 
                std=torch.tensor([0.2023, 0.1994, 0.2010])),
    data_keys=['input']
)

test_transformation = K.AugmentationSequential(
    # 1. Normalization
    K.Normalize(mean=torch.tensor([0.4914, 0.4822, 0.4465]), 
                std=torch.tensor([0.2023, 0.1994, 0.2010])),
    data_keys=['input']
)

raw_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=base_transformation, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=base_transformation, download=True)

classes = full_train_dataset.classes

# split data into train and val
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
# torch generator
generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=generator)

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# visualize train or val data
visualize_smpl(train_dataset, classes, 'train')
visualize_smpl(val_dataset, classes, 'val')

if __name__ == "__main__":
    get_device()
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test Samples: {len(test_dataset)}")