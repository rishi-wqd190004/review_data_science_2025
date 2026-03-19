import torch
import torchvision
import kornia.augmentation as K

def get_device():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu_name} (count: {torch.cuda.device_count()})")
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon)")
        return torch.device('mps')
    else:
        print("Using CPU")
        return torch.device('cpu')

train_transformation = K.AugmentationSequential(
    # 1. Geometric
    K.RandomHorizontalFlip(p=0.5),
    K.RandomCrop((32,32), padding=4, p=1.0),
    K.RandomRotation(degrees=15.0, p=0.5),
    # 2. Photometric
    K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    # 3. Noise/Regularization
    K.RandomGaussianNoise(mean=0.0, std=0.01, p=0.2),
    # 4. Normalization
    K.Normalize(mean=torch.tensor([0.4914, 0.4822, 0.4465]), 
                std=torch.tensor([0.2023, 0.1994, 0.2010])),
    data_keys=['input']
)

val_transformation = K.AugmentationSequential(
    # 1. Normalization
    K.Normalize(mean=torch.tensor([0.4914, 0.4822, 0.4465]), 
                std=torch.tensor([0.2023, 0.1994, 0.2010])),
    data_keys=['input']
)

raw_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=train_transformation, download=True)
val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=val_transformation, download=True)


if __name__ == "__main__":
    get_device()