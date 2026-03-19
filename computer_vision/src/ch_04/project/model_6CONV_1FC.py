from torch import nn
from torchinfo import summary

class MDL_6CONV_1FC(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            # conv2
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            # Pool + dropout
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2),

            # conv3
            nn.Conv2d(in_channels=32, out_channels=32*2, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32*2),

            # conv4
            nn.Conv2d(in_channels=32*2, out_channels=32*2, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32*2),

            # Pool + dropout
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.3),

            # conv5
            nn.Conv2d(in_channels=32*2, out_channels=32*4, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32*4),

            # conv6
            nn.Conv2d(in_channels=32*4, out_channels=32*4, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32*4),

            # Pool + dropout
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.4),

            # FC
            nn.Flatten(),
            nn.Linear(in_features=128*4*4, out_features=10)
        )

    def forward(self, x):
        return self.layer_stack(x)
    

mdl = MDL_6CONV_1FC()
summary(mdl, input_size=(1,3,32,32))