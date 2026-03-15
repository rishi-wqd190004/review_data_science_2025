import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from data_aug_101 import raw_dataset


image_pil = raw_dataset[0][0]
image_np = np.array(image_pil)

# albumentation pipeline
albumentations_transform = A.Compose([
    A.HorizontalFlip(p=1.0),
    A.RandomBrightnessContrast(p=1.0),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=1.0),
    ToTensorV2()
])

# apply augmentation
augmented = albumentations_transform(image=image_np)
img_alb = augmented['image'] # CHW format

# convert tensor back to HWC np visualization
img_alb_np = img_alb.permute(1,2,0).numpy()

# albumentations op is uint8 range, if its off then multiple by 255 as TotensorV2() normalizes to [0,1]
if img_alb_np.max() <= 1.0:
    img_alb_np = (img_alb_np * 255).astype(np.uint8)

# Visualize original and augmented images side-by-side
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(image_np)
axs[0].set_title('Original')
axs[0].axis('off')
axs[1].imshow(img_alb_np)
axs[1].set_title('Albumentations')
axs[1].axis('off')
plt.tight_layout()
plt.savefig('albumentation.jpg')
plt.show()