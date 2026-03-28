import os
import webdataset as wds
from tqdm import tqdm

def get_class_mapping(base_path):
    """Creates a mapping from WordNet ID (nXXXX) to 0-999 index based on Train folders."""
    # Note: Adjust path to match your ILSVRC structure
    train_dir = os.path.join(base_path, "Data/CLS-LOC/train")
    wnids = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    return {wnid: i for i, wnid in enumerate(wnids)}

def shard_split(base_path, mode, output_dir, is_flat=False, is_test=False):
    input_dir = os.path.join(base_path, f"Data/CLS-LOC/{mode}")
    output_pattern = os.path.join(output_dir, f"{mode}-%06d.tar")
    class_mapping = get_class_mapping(base_path)
    
    samples = []
    if not is_flat:
        # TRAIN: Walk subfolders
        for wnid, idx in class_mapping.items():
            folder_path = os.path.join(input_dir, wnid)
            if os.path.exists(folder_path):
                for f in os.listdir(folder_path):
                    if f.lower().endswith(('.jpeg', '.jpg')):
                        samples.append((os.path.join(folder_path, f), idx))
    elif is_test:
        # TEST: Dummy labels
        files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.jpeg')])
        samples = [(os.path.join(input_dir, f), -1) for f in files]
    else:
        # VAL: Flat folder + Ground Truth Mapping
        val_map_file = os.path.join(base_path, "ImageSets/CLS-LOC/val.txt")
        with open(val_map_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # Your val.txt is "filename label"
                img_name = parts[0] + ".JPEG" 
                label = int(parts[1]) - 1 # Convert 1-1000 to 0-999
                samples.append((os.path.join(input_dir, img_name), label))

    print(f"Starting {mode} sharding: {len(samples)} images.")

    with wds.ShardWriter(output_pattern, maxsize=1e9 if mode=="train" else 5e8) as sink:
        for i, (path, label) in enumerate(tqdm(samples, desc=f"Writing {mode}")):
            if not os.path.exists(path): continue
            with open(path, 'rb') as f:
                img_data = f.read()
            
            sink.write({
                "__key__": f"{mode}_{i:08d}",
                "jpg": img_data,
                "cls": label,
            })

if __name__ == "__main__":
    # SETTINGS: Point to the ILSVRC root
    base = "/media/rishi/shared/ubuntu_data/review_data_science_2025/datasets/imagenet/ILSVRC"
    out = "/media/rishi/shared/ubuntu_data/review_data_science_2025/datasets/imagenet_shards"
    
    os.makedirs(out, exist_ok=True)
    
    # 1. Shard Val first (it's fast and will prove if the CUDA error is gone)
    shard_split(base, "val", out, is_flat=True)
    
    # 2. Shard Train (The long process)
    shard_split(base, "train", out, is_flat=False)
    
    # 3. Shard Test
    shard_split(base, "test", out, is_flat=True, is_test=True)