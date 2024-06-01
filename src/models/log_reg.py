import torch
from lightning import LightningModule
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score

from src.data.dataset import NUM_CLASSES


class MulticlassLogisticRegression(LightningModule):
    def __init__(self, input_dim, num_classes=NUM_CLASSES):
        super(MulticlassLogisticRegression, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, num_classes)
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.f1 = F1Score(num_classes=num_classes, average='weighted', task='multiclass')

    def forward(self, x):
        x = x.view(x.size(0), -1)
        assert x.size(1) == self.input_dim, f"Input dimension {x.size(1)} does not match expected {self.input_dim}"

        return self.linear(x)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def _shared_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        y = torch.argmax(y, dim=1)
        loss = F.cross_entropy(logits, y)
        acc = self.accuracy(preds, y)
        f1 = self.f1(preds, y)
        return loss, acc, f1

    def training_step(self, batch, batch_idx):
        loss, acc, f1 = self._shared_step(batch, batch_idx)
        self.log_dict({"train_loss": loss, "train_acc": acc, "train_f1": f1}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, f1 = self._shared_step(batch, batch_idx)
        self.log_dict({"val_loss": loss, "val_acc": acc, "val_f1": f1}, prog_bar=True)