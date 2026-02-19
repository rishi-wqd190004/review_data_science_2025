import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import random_split

generator = torch.Generator().manual_seed(42)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(
         mean=(0.5,0.5,0.5),
         std=(0.5,0.5,0.5)
     )]
)
batch_size = 4

full_set = CIFAR10(
                root='./data',
                train=True,
                download=True,
                transform=transform
            )

train_size = int(0.8 * len(full_set))
val_size = len(full_set) - train_size

train_set, val_set = random_split(full_set, [train_size, val_size], generator=generator)

trainloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=2)

valloader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, num_workers=2)

testset = CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transform
            )
testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
