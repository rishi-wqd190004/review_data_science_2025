import lightning as L
import torch
import torch.nn as nn
import torchmetrics

class CIFAR10_6CONV_1FC_LightTrain(L.LightningModule):
    def __init__(self, model, train_trans, val_trans, num_classes=10, lr=1e-3):
        super().__init__()
        self.model = model
        self.train_trans = train_trans
        self.val_trans = val_trans
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

        # torchmetrics for accuracy and f1
        self.train_metrics = torchmetrics.MetricCollection(
            {
                "accuracy": torchmetrics.classification.Accuracy(task='multiclass', num_classes=num_classes),
                "f1": torchmetrics.classification.F1(task='multiclass', num_classes=num_classes)
            },
            prefix='train_',
        )
        self.val_metrics = self.train_metrics.clone(prefix='val_')

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.train_trans(x)
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.train_metrics(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log_dict(self.train_metrics, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_batch_end(self):
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.val_trans(x)
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.val_metrics(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(self.val_metrics, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)