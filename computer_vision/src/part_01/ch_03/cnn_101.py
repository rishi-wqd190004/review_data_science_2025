from torch import nn
from torchinfo import summary

class CNN101(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            
            nn.Linear(in_features=7*7*64, out_features=64),
            nn.ReLU(),

            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        return self.layer_stack(x)
    
cnn_101_mdl = CNN101()
summary(cnn_101_mdl, input_size=(1,1,28,28))