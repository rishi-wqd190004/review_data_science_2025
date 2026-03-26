import os
import webdataset as wds
from torchvision.datasets import ImageFolder
from tqdm import tqdm

def shard_ds(base_path, output_dir, splits=["train", "val", "test"]):
    os.makedirs(output_dir, exist_ok=True)

    for mode in splits:
        input_dir = os.path.join(base_path, mode)
        if not os.path.exists(input_dir):
            print(f"Skipping {mode}: Directory not found at {input_dir}")
            continue

        output_pattern = os.path.join(output_dir, f"{mode}-%06d.tar")

        # check subdirectories exist
        subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

        if len(subdirs) > 0:
            print(f"Detected categorized structure for {mode}. Using ImageFolder logic...")
            ds = ImageFolder(input_dir)
            samples = ds.imgs
            class_names = ds.classes
        else:
            print(f"Detected flat structure for {mode}. Assigning dummy labels...")
            files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                           if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
            samples = [(f, -1) for f in files]
            class_names = ["unknown"]
        print(f"Sharding {mode}: {len(samples)} images found.")

        # shard size
        max_size = 1e9 if mode == "train" else 5e8

        with wds.ShardWriter(output_pattern, maxsize=max_size) as sink:
            for i, (path, label) in enumerate(tqdm(samples, desc=f"Writing {mode}")):
                with open(path, 'rb') as f:
                    img_data = f.read()

                # unique keyness: split_index_filename
                fname = os.path.basename(path)
                key = f"{mode}_{i:08d}_{fname}"

                sink.write({
                    "__key__": key,
                    "jpg": img_data,
                    "cls": label,
                })

# settings
base_path = "/media/rishi/shared/ubuntu_data/review_data_science_2025/datasets/imagenet/ILSVRC/Data/CLS-LOC"
output_dir = "/media/rishi/shared/ubuntu_data/review_data_science_2025/datasets/imagenet_shards"
output_pattern = os.path.join(output_dir, "train-%06d.tar")

shard_ds(base_path, output_dir)