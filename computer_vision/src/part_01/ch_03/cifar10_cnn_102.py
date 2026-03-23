import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import random_split

generator = torch.Generator().manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example with MPS (Apple Silicon) support
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(
         mean=(0.5,0.5,0.5),
         std=(0.5,0.5,0.5)
     )]
)
batch_size = 64

full_set = CIFAR10(
                root='./data',
                train=True,
                download=True,
                transform=transform
            )

train_size = int(0.8 * len(full_set))
val_size = len(full_set) - train_size

train_set, val_set = random_split(full_set, [train_size, val_size], generator=generator)

trainloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)

valloader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, num_workers=0)

testset = CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transform
            )
testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def train_step(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, accuracy_metric, device: torch.device=device):
    model.train()
    train_loss = 0
    accuracy_metric.reset()
    for batch, (x,y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        # 1. forward pass
        y_pred = model(x)
        # 2. calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        # 3. optimizer zero grad
        optimizer.zero_grad()
        # 4. loss backward
        loss.backward()
        # 5. optimizer step
        optimizer.step()

        preds = torch.argmax(y_pred, dim=1)
        accuracy_metric.update(preds, y)

    # per epoch train loss and accuracy
    train_loss /= len(data_loader)
    train_acc = accuracy_metric.compute()

    return train_loss, train_acc

def test_step(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, accuracy_metric, device: torch.device=device, valid: bool=True):
    test_loss = 0
    accuracy_metric.reset()
    model.eval()
    # inference context manager
    with torch.inference_mode():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            # 1. forward pass
            test_pred = model(x)
            # 2. train and loss accuracy
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()
            preds = torch.argmax(test_pred, dim=1)
            accuracy_metric.update(preds, y)

        # per epoch train loss and accuracy
        test_loss /= len(data_loader)
        test_acc = accuracy_metric.compute()
        if valid:
            return test_loss, test_acc
        else:
            return test_loss, test_acc