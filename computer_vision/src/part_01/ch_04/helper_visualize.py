import matplotlib.pyplot as plt
import numpy as np

def unnormalize(img_tensor):
    # convert normalized tensor back to image
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    img = img_tensor.numpy().transpose((1,2,0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def show_augmented_vs_org(transform, dataset, num_images=8):
    # get original image PIL from dataset
    orignal_images = [dataset[i][0] for i in range(num_images)]

    # apply augmentation
    augmented_imgs = [transform(img) for img in orignal_images]

    fig, axis = plt.subplots(2, num_images, figsize=(16,4))
    fig.suptitle("Original images vs augmented images", fontsize=16)

    for i in range(num_images):
        axis[0, i].imshow(orignal_images[i])
        axis[0, i].axis('off')

        if i == 0:
            axis[0, i].set_ylabel('Original', fontsize=12)

        # display augmented image
        img = unnormalize(augmented_imgs[i])
        axis[1, i].imshow(img)
        axis[1, i].axis('off')

        if i == 0:
            axis[0, i].set_ylabel('Augmented', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('augmentation_check.jpg')
    plt.show()