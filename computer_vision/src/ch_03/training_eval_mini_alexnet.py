import torch
from torch import nn
from tqdm import tqdm
from mini_alexnet import MiniALEXNET
from torchinfo import summary
from cifar10_cnn_102 import device, train_step, test_step, trainloader, valloader, testloader, classes
from torchmetrics.classification import MulticlassAccuracy
from timeit import default_timer as timer
import json
import os
os.makedirs("model", exist_ok=True)

mini_alex = MiniALEXNET().to(device)
summary(mini_alex, input_size=(1, 3, 32, 32))

train_acc_metric = MulticlassAccuracy(num_classes=len(classes)).to(device)
val_acc_metric = MulticlassAccuracy(num_classes=len(classes)).to(device)
test_acc_metric = MulticlassAccuracy(num_classes=len(classes)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=mini_alex.parameters(), lr=0.01)

# lr_scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.1,
    patience=2
)

# training
batch_size = 64
epochs = 100
train_accs, val_accs, test_accs = [], [], []

# early stopping variables
best_val_acc = 0
early_stopping_patience = 5
epochs_without_improvement = 0

# timer
train_time_start_mini_alex = timer()
pbar = tqdm(range(epochs), desc="Training Progress")
for epoch in pbar:
    tqdm.write(f"Epoch: {epoch}")
    tqdm.write("------")
    train_loss, train_acc = train_step(
                            data_loader=trainloader,
                            model=mini_alex,
                            loss_fn=loss_fn,
                            optimizer=optimizer,
                            accuracy_metric=train_acc_metric
                        )
    train_acc = train_acc_metric.compute().item()
    train_acc_metric.reset()
    train_accs.append(train_acc)

    test_loss, test_acc = test_step(
                            data_loader=valloader,
                            model=mini_alex,
                            loss_fn=loss_fn,
                            accuracy_metric=val_acc_metric,
                            valid=True
                        )
    test_acc = val_acc_metric.compute().item()
    val_acc_metric.reset()
    val_accs.append(test_acc)
    # scheduler step
    scheduler.step(test_acc)

    # eraly stopping
    if test_acc > best_val_acc:
        best_val_acc = test_acc
        epochs_without_improvement = 0
        torch.save(mini_alex.state_dict(), "model/best_model.pth")
    else:
        epochs_without_improvement += 1
    
    if epochs_without_improvement >= early_stopping_patience:
        print("Early stopping triggered")
        break
    pbar.set_postfix({
        "train_loss": f"{train_loss:.4f}",
        "train_acc": f"{train_acc:.4f}",
        "val_loss": f"{test_loss:.4f}",
        "val_acc": f"{test_acc:.4f}",
        "lr": optimizer.param_groups[0]["lr"]
    })
    tqdm.write(f"\nEpoch [{epoch+1}/{epochs}]")
    tqdm.write(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    tqdm.write(f"Val   Loss: {test_loss:.4f} | Val   Acc: {test_acc:.4f}")
    tqdm.write("-" * 50)

train_time_stop_mini_alex = timer()
total_train_time_model = (train_time_stop_mini_alex - train_time_start_mini_alex) / 60
print(f"Total model training and validation time: {total_train_time_model:6f} minutes")

# load the best model for prediction
mini_alex.load_state_dict(torch.load("model/best_model.pth"))
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

print(type(train_accs[0]))
print(type(val_accs[0]))
print(type(test_accs[0]))
results_accuracies = {
    "train_accs": train_accs,
    "val_accs": val_accs,
    "test_accs": test_accs
}
with open("results_accuracies.json", "w") as f:
    json.dump(results_accuracies, f)

print("Training, Validation and Test complete. Accuracies written into json file")