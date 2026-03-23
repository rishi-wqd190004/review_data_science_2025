## Data Augmentation

### Torchvision.transforms and transforms.v2
 
- Does the basic augmentation

### Albumentations

- It applies augmentation with variations like Affine, HorizontalFlip, etc. In other words, more refined.

- You can control **p** for each of the image or each segment of the transformation pipeline.

### Kornia

- Applies transformation in N,C,H,W format.

- Speed on large batches/GPUs