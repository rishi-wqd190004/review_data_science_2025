import torch
from torch.utils.data import Subset, DataLoader
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from transform_subset import TransformSubset

print(f"Device: {torch.cuda.get_device_name(0)}")
BATCH_SIZE = 64
NUM_WORKERS = 1

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((226)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


full_dataset = datasets.ImageFolder(root='./data/imagenet1kv1')

indices = list(range(len(full_dataset)))
labels = full_dataset.targets

train_idx, rest_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)

val_idx, test_idx = train_test_split(rest_idx, test_size=0.5, stratify=[labels[i] for i in rest_idx], random_state=42)

# use custom subset instead of Subset
train_data = TransformSubset(Subset(full_dataset, train_idx), transform=train_transform)
val_data = TransformSubset(Subset(full_dataset, val_idx), transform=val_test_transform)
test_data = TransformSubset(Subset(full_dataset, test_idx), transform=val_test_transform)

# data loader
train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)