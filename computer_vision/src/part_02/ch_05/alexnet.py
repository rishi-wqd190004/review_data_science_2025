import torch
from torch import nn
from torchinfo import summary

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # convolutional layers for feature extraction
        self.layer_stack = nn.Sequential(
            # conv1 + pool + batchnorm
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # conv2 + pool + batchnorm
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            

            # conv3 + batchnorm
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(inplace=True),
            
            # conv4 + batchnorm
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(inplace=True),
            
            # conv5 + batchnorm
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # fc
            nn.Flatten(),
            nn.Linear(in_features=6*6*256, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=4096, out_features=num_classes),
        )
    def forward(self, x):
        return self.layer_stack(x)
    
mdl = AlexNet()
summary(mdl, input_size=(1,3,227,227))