import torch
import matplotlib.pyplot as plt
from computer_vision.src.part_02.ch_05.alexnet.wds_loader import get_wds_loader, ImageNetAugmentor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
aug = ImageNetAugmentor(mode="train").to(device)
loader = get_wds_loader("train", "/media/rishi/shared/ubuntu_data/review_data_science_2025/datasets/imagenet_shards", batch_size=4)

# grab 1 batch
batch = next(iter(loader))
images, labels = batch
images = images.to(device)

# apply aug
with torch.no_grad():
    aug_imgs = aug(images)

# denormalize the images (reverse the mean and std) 
# imgs will look dark and weird
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
vis_images = (aug_imgs * std) + mean

# plot
fig, axes = plt.subplots(2, 4, figsize=(15, 8))
for i in range(4):
    axes[0, i].imshow(images[i].cpu().permute(1, 2, 0))
    axes[0, i].set_title(f"Original (Class {labels[i]})")
    
    # Clamp to 0-1 for matplotlib safety
    img_to_show = vis_images[i].cpu().permute(1, 2, 0).clamp(0, 1)
    axes[1, i].imshow(img_to_show)
    axes[1, i].set_title("After Augmentor")

plt.tight_layout()
plt.savefig("check_my_images.png")
print("Check 'check_my_images.png' now!")