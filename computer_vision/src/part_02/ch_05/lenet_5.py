import torch
from torch import nn
from torchinfo import summary

class LeNet_5(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
            # conv1
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # conv2
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # conv3
            nn.Conv2d(16, 120, kernel_size=5),

            # fc1
            nn.Flatten(),

            # fc2
            nn.Linear(120, 84),
            nn.Tanh(),
            # fc3
            nn.Linear(84, 10)
        )
    
    def forward(self, x):
        return self.layer_stack(x)
    
mdl = LeNet_5()
summary(mdl, input_size=(1,1,28,28))