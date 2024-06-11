import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import F1Score, Accuracy
import torch.nn.functional as F


class Simple1DCNN(LightningModule):
    def __init__(self, optimizer, input_length, input_channels, num_classes):
        super().__init__()
        self.save_hyperparameters()

        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(64 * (input_length // 8), 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.5)

        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.f1_score = F1Score(num_classes=num_classes, average='weighted', task='multiclass')

    def forward(self, x):
        # switch from (batch, channels, length) to (batch, length, channels)
        x = x.permute(0, 2, 1)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def _shared_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        y = torch.argmax(y, dim=1)
        loss = F.cross_entropy(logits, y)
        acc = self.accuracy(preds, y)
        f1 = self.f1_score(preds, y)
        return loss, acc, f1

    def training_step(self, batch, batch_idx):
        loss, acc, f1 = self._shared_step(batch, batch_idx)
        self.log_dict({"train_loss": loss, "train_acc": acc, "train_f1": f1}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, f1 = self._shared_step(batch, batch_idx)
        self.log_dict({"val_loss": loss, "val_acc": acc, "val_f1": f1}, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return preds

    def configure_optimizers(self):
        return self.hparams.optimizer(params=self.trainer.model.parameters())