import torch
from data_preprocessing import *
from model_6CONV_1FC import *
from train_loop import *
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

logger = TensorBoardLogger("tb_logs", name="cifar10_6conv_1fc")

def main():
    device = get_device()

    raw_mdl = MDL_6CONV_1FC()

    # initialize model
    lit_model = CIFAR10_6CONV_1FC_LightTrain(
        model=raw_mdl,
        train_trans=train_transformation,
        val_trans= test_transformation
    )
    # setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="checkpoints/",
        filename="cifar10-{epoch:02d}-{val_acc:.2f}",
        save_top_k=1,
        mode="max"
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min"
    )

    # initialize the trainer
    trainer = Trainer(
        max_epochs=100,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        precision="16-mixed", # using half precision as faster on modern GPUs
        log_every_n_steps=10
    )

    # start training
    trainer.fit(
        model=lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

if __name__ == "__main__":
    main()