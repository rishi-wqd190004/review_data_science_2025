import torch
from torchinfo import summary
from torch import nn
from tqdm.auto import tqdm
from timeit import default_timer as timer
from cifar10_cnn_102 import device, train_step, test_step, trainloader, valloader, testloader, classes
from torchmetrics.classification import MulticlassAccuracy

train_acc_metric = MulticlassAccuracy(num_classes=len(classes)).to(device)
val_acc_metric = MulticlassAccuracy(num_classes=len(classes)).to(device)

class MiniALEXNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Dropout2d(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(in_features=64*4*4, out_features=500),
            nn.ReLU(),

            nn.Dropout2d(0.3),

            nn.Linear(in_features=500, out_features=10),
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.classifier(x)
        return x
    
mini_alex = MiniALEXNET()
summary(mini_alex, input_size=(1, 3, 32, 32))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=mini_alex.parameters(), lr=0.01)

# timer
train_time_start_mini_alex = timer()
# training
epochs = 10
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n------")
    train_step(
        data_loader=trainloader,
        model=mini_alex,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_metric=train_acc_metric
    )
    test_step(
        data_loader=valloader,
        model=mini_alex,
        loss_fn=loss_fn,
        accuracy_metric=val_acc_metric,
        valid=True
    )
    # test_step(
    #     data_loader=testloader,
    #     model=mini_alex,
    #     loss_fn=loss_fn,
    #     accuracy_fn=accuracy_fn,
    #     valid=False
    # )
train_time_stop_mini_alex = timer()
total_train_time_model = train_time_stop_mini_alex - train_time_start_mini_alex
print(total_train_time_model)