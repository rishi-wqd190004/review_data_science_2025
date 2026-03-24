# this is latest version of computer vision using devs from torchvision
import torch
from torch import nn
from torchinfo import summary

class AlexNet_V2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # 1. convolutional layer for feature extraction 5
        self.feature_selection = nn.Sequential(
            # layer 1
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # layer 2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # layer 3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # layer 4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # layer 5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # 2. adaptive pooling to get output as 6x6
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))

        # 3. Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(6*6*256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.feature_selection(x)
        x = self.avgpool(x)
        # flatten the 3D tensor into 1D vector for linear layers
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
mdl = AlexNet_V2()
summary(mdl, input_size=(1,3,224,224))