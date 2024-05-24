import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy, F1Score


class MulticlassLogisticRegression(LightningModule):
    def __init__(self, input_dim, num_classes):
        super(MulticlassLogisticRegression, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, num_classes)
        self.accuracy = Accuracy(num_classes=num_classes, task='multiclass')
        self.f1 = F1Score(num_classes=num_classes, task='multiclass')

    def forward(self, x):
        # Flatten the input data if it's not already
        x = x.view(x.size(0), -1)
        assert x.size(1) == self.input_dim, f"Input dimension {x.size(1)} does not match expected {self.input_dim}"
        return self.linear(x)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.argmax(dim=1)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = self.accuracy(y_hat.argmax(dim=1), y)
        f1 = self.f1(y_hat.argmax(dim=1), y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_f1', f1, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.argmax(dim=1)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = self.accuracy(y_hat.argmax(dim=1), y)
        f1 = self.f1(y_hat.argmax(dim=1), y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = self.accuracy(y_hat.argmax(dim=1), y)
        f1 = self.f1(y_hat.argmax(dim=1), y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_f1', f1, on_step=True, on_epoch=True, prog_bar=True)
        return loss
