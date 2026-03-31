import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from computer_vision.src.part_02.ch_05.alexnet.alexnet_v2 import AlexNet_V2
from computer_vision.src.part_02.ch_05.alexnet.wds_loader import get_wds_loader, ImageNetAugmentor

class ImageNetLightningTrainer(pl.LightningModule):
    def __init__(self, model, train_path, val_path, num_classes=1000, batch_size=128):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        # GPU augmentation
        self.aug_train = ImageNetAugmentor(mode="train")
        self.aug_val = ImageNetAugmentor(mode='val')
        self.aug_test = ImageNetAugmentor(mode='test')

        # metrics using torchmetrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

        #self.conf_mat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, targets = batch

        # DEBUG: Check if any label is out of bounds
        if targets.max() >= 1000 or targets.min() < 0:
            print(f"!!! CRITICAL LABEL ERROR !!! Max: {targets.max()}, Min: {targets.min()}")
            # This will stop the script before the GPU crashes
            raise ValueError("Target label out of range for 1000 classes.")

        # apply gpu aug
        images = self.aug_train(images)

        outputs = self(images)
        loss = self.criterion(outputs, targets)
        self.train_acc(outputs, targets)

        # log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # HARD FILTER: Only allow 0-999
        # If any label is 1000 or -1, this removes it from the batch
        mask = (targets >= 0) & (targets < 1000)
        if not mask.any():
            return None # Skip if the whole batch is garbage
        images = images[mask]
        targets = targets[mask]

        images = self.aug_val(images)

        outputs = self(images)
        loss = self.criterion(outputs, targets)

        # calculated metrics
        self.val_acc(outputs, targets)
        self.val_f1(outputs, targets)
        #self.conf_mat(outputs, targets)

        # log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True)
        return loss
    
    # def on_validation_batch_end(self):
    #     pass
    #     # cm = self.conf_mat.compute()
    #     # self.conf_mat.reset()

    def configure_optimizers(self):
        steps_per_epoch = 1281167 // self.hparams.batch_size
        total_steps = steps_per_epoch * self.trainer.max_epochs
        optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.1,
            total_steps=total_steps,#self.trainer.estimated_stepping_batches,
            pct_start=0.1
        )
        return {
            'optimizer': optimizer,#optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3),
            'lr_scheduler': {'scheduler': scheduler, "interval": "step"}
        }
        #return [optimizer], [scheduler]
    
    def train_dataloader(self):
        return get_wds_loader("train", self.hparams.train_path, batch_size=self.hparams.batch_size)
    
    def val_dataloader(self):
        return get_wds_loader("val", self.hparams.val_path, batch_size=self.hparams.batch_size)
    

def main():
    # path
    SHARD_PATH = "/media/rishi/shared/ubuntu_data/review_data_science_2025/datasets/imagenet_shards"
    CHECKPOINT_DIR = "./checkpoints"

    alexnet = AlexNet_V2(num_classes=1000)

    system = ImageNetLightningTrainer(
        model=alexnet,
        train_path=SHARD_PATH,
        val_path=SHARD_PATH,
        batch_size=512
    )

    # callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="alexnet_v2_{epoch:02d}_{val_acc:.2f}",
        dirpath=CHECKPOINT_DIR,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # trainer config
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=200,
        precision="bf16-mixed",
        benchmark=True,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=25,
    )

    print("starting Training")
    trainer.fit(system)

if __name__ == "__main__":
    main()