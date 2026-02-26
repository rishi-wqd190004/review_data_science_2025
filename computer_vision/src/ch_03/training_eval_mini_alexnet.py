import torch
from torch import nn
from tqdm import tqdm
from mini_alexnet import MiniALEXNET
from torchinfo import summary
from cifar10_cnn_102 import device, train_step, test_step, trainloader, valloader, testloader, classes
from torchmetrics.classification import MulticlassAccuracy
from timeit import default_timer as timer

mini_alex = MiniALEXNET().to(device)
summary(mini_alex, input_size=(1, 3, 32, 32))

train_acc_metric = MulticlassAccuracy(num_classes=len(classes)).to(device)
val_acc_metric = MulticlassAccuracy(num_classes=len(classes)).to(device)
test_acc_metric = MulticlassAccuracy(num_classes=len(classes)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=mini_alex.parameters(), lr=0.01)

# timer
train_time_start_mini_alex = timer()
# training
epochs = 10
train_accs, val_accs, test_accs = [], [], []

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n------")
    train_step(
        data_loader=trainloader,
        model=mini_alex,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_metric=train_acc_metric
    )
    train_acc = train_acc_metric.compute().item()
    train_acc_metric.reset()
    train_accs.append(train_acc)

    test_step(
        data_loader=valloader,
        model=mini_alex,
        loss_fn=loss_fn,
        accuracy_metric=val_acc_metric,
        valid=True
    )
    val_acc = val_acc_metric.compute().item()
    val_acc_metric.reset()
    val_accs.append(val_acc)

train_time_stop_mini_alex = timer()
total_train_time_model = train_time_stop_mini_alex - train_time_start_mini_alex
print(total_train_time_model)
print("************")
print(train_accs)

test_step(
    data_loader=testloader,
    model=mini_alex,
    loss_fn=loss_fn,
    accuracy_metric=test_acc_metric,
    valid=False
)
test_acc = test_acc_metric.compute().item()
test_acc_metric.reset()
test_accs.append(test_acc)