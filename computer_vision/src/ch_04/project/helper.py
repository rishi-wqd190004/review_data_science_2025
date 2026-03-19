import matplotlib.pyplot as plt
import torch

def visualize_smpl(train_dataset, classes, file_name:str, rows=3, cols=3):
    figure = plt.figure(figsize=(8,8))
    for i in range(1, cols * rows +1):
        sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
        img, label_idx = train_dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        ax_title = f"Label: {classes[label_idx]}"
        plt.title(ax_title)
        plt.axis("off")
        plt.imshow(img.permute(1,2,0).detach().cpu(), cmap="gray")
    plt.savefig(f"sample_{file_name}_.jpg")