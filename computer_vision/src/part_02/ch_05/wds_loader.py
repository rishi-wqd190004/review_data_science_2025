import torchvision.transforms as T
import webdataset as wds
import os, glob
from torch.utils.data import DataLoader

def get_wds_loader(mode, shard_path, batch_size=256, ssd_cache="/home/rishi/review_data_science_2025/computer_vision/src/part_02/ch_05/.cache/wds"):

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
            shardshuffle=(mode=="train")
            )
            .shuffle(1000 if mode == "train" else 0)
            .decode("pil")
            .to_tuple("jpeg.jpg", "jpeg.cls") # as per the data_preprocessing tar files
            .map_tuple(base_transform, lambda y: y)
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
        drop_last=(mode == "train")
    )

    return loader

shard_path = "/media/rishi/shared/ubuntu_data/review_data_science_2025/datasets/imagenet_shards"
train_loader = get_wds_loader("train", shard_path)
val_loader = get_wds_loader("val", shard_path)
test_loader = get_wds_loader("test", shard_path)
print("DataLoader done")
images, targets = next(iter(train_loader))
print(images.shape)