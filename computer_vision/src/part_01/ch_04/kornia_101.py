import numpy as np
import torch
import torchvision
import kornia.augmentation as K
import matplotlib.pyplot as plt
from data_aug_101 import raw_dataset

def tensor_to_np(img_tensor):
    # convert CHW (0-1) to HWC uint8 for matplotlib
    img_np = img_tensor.permute(1,2,0).numpy()
    img_np = np.clip(img_np * 255,0,255).astype(np.uint8)
    return img_np

img_pil, label = raw_dataset[0]

# converting PIL image to tensor
img_tensor = torchvision.transforms.ToTensor()(img_pil)

# add batch dimension (1,c,h,w)
img_tensor_batch = img_tensor.unsqueeze(0)

# define kornia augmentation pipeline
korina_transform = K.AugmentationSequential(
    K.RandomHorizontalFlip(p=1.0),
    K.RandomRotation(degrees=30.0),
    K.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
)

# apply augmentation
img_korina_batch = korina_transform(img_tensor_batch)

# remove batch dimension
img_korina = img_korina_batch.squeeze(0)

# convert to np
img_orig_np = tensor_to_np(img_tensor)
img_aug_np = tensor_to_np(img_korina)

# Plot side by side
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img_orig_np)
axs[0].set_title('Original')
axs[0].axis('off')

axs[1].imshow(img_aug_np)
axs[1].set_title('Kornia Augmented')
axs[1].axis('off')

plt.tight_layout()
plt.savefig('korina_aug.jpg')
plt.show()