import torch
from torch.utils.data import Subset
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(root='./data/imagenet1kv1')

indices = list(range(len(full_dataset)))
labels = full_dataset.targets

train_idx, rest_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)

val_idx, test_idx = train_test_split(rest_idx, test_size=0.5, stratify=[labels[i] for i in rest_idx], random_state=42)

train_data = Subset(full_dataset, train_idx)
val_data = Subset(full_dataset, val_idx)
test_data = Subset(full_dataset, test_idx)

print(len(train_data))