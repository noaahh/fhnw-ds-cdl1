import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import Accuracy, F1Score


class SimpleLSTM(LightningModule):
    def __init__(self, optimizer, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.accuracy = Accuracy(task='multiclass', num_classes=output_dim)
        self.f1_score = F1Score(num_classes=output_dim, average='weighted', task='multiclass')

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # take the output of the last time step
        out = self.fc(lstm_out)
        return out

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

    def configure_optimizers(self):
        return self.hparams.optimizer(params=self.trainer.model.parameters())
