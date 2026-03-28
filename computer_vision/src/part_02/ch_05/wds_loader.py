import torchvision.transforms as T
import webdataset as wds
import os, glob
import torch
import torch.nn as nn
import kornia as K
from torch.utils.data import DataLoader

def get_wds_loader(mode, shard_path, batch_size=64, ssd_cache="/home/rishi/review_data_science_2025/computer_vision/src/part_02/ch_05/.cache/wds"):

    pattern = os.path.join(shard_path, f"{mode}-*.tar")
    files = sorted(glob.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No shards found for {mode} in {shard_path}")
    
    os.makedirs(ssd_cache, exist_ok=True)
    # simple crop with 256 so GPU can resize further
    base_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])

    dataset = (
        wds.WebDataset(
            files,
            cache_dir=ssd_cache,
            shardshuffle=True
            )
            .shuffle(1000 if mode == "train" else 0)
            .decode("pil")
            .to_tuple("jpg", "cls") # as per the data_preprocessing tar files
            .map_tuple(base_transform, lambda y: y)
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=(mode == "train")
    )

    return loader

# image augementor class
class ImageNetAugmentor(nn.Module):
    def __init__(self, mode="train"):
        super().__init__()
        self.mode = mode

        # common normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])

        if self.mode == "train":
            # training- randomness for generalization
            self.transform = nn.Sequential(
                K.augmentation.RandomResizedCrop(size=(224,224), scale=(0.08, 1.0)),
                K.augmentation.RandomHorizontalFlip(),
                K.augmentation.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                K.augmentation.Normalize(mean=self.mean, std=self.std)
            )
        else:
            # val/test - deterministic for consistency
            self.transform = nn.Sequential(
                K.augmentation.CenterCrop(size=(224,224)),
                K.augmentation.Normalize(mean=self.mean, std=self.std)
            )
    
    @torch.no_grad()
    def forward(self, x):
        return self.transform(x)

def main():
    shard_path = "/media/rishi/shared/ubuntu_data/review_data_science_2025/datasets/imagenet_shards"
    train_loader = get_wds_loader("train", shard_path)
    val_loader = get_wds_loader("val", shard_path)
    test_loader = get_wds_loader("test", shard_path)
    print("DataLoader done")
    images, targets = next(iter(train_loader))
    print(images.shape)

if __name__ == "__main__":
    #main()
    pass