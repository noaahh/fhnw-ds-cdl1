import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import Accuracy, F1Score


class BidirectionalLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x


class ResidualLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, num_blocks):
        super().__init__()
        self.first_linear = nn.Linear(input_dim, input_dim)
        output_dim = hidden_dim * 2
        self.layers = nn.ModuleList()
        for i in range(num_blocks):
            self.layers.append(
                nn.Sequential(
                    BidirectionalLayer(input_dim, hidden_dim, num_layers, dropout),
                    nn.Linear(output_dim, input_dim),
                    nn.ReLU(),
                )
            )
        self.batch_norm = nn.BatchNorm1d(input_dim)

    def forward(self, x):
        x = self.first_linear(x)
        x = F.relu(x)

        residual = None
        for i, layer in enumerate(self.layers):
            layer_output = layer(x)
            if i != 0:
                x = layer_output + residual
            else:
                x = layer_output
            residual = x

        x = x.transpose(1, 2)
        x = self.batch_norm(x)
        x = x.transpose(1, 2)
        return x


class DeepResBidirLSTM(LightningModule):
    def __init__(self, optimizer, input_dim, hidden_dim, num_layers, dropout, num_blocks, output_dim):
        super().__init__()
        self.save_hyperparameters()
        self.residual_layer = ResidualLayer(input_dim, hidden_dim, num_layers, dropout, num_blocks)
        self.final_fc = nn.Linear(input_dim, output_dim)
        self.accuracy = Accuracy(task='multiclass', num_classes=output_dim)
        self.f1_score = F1Score(num_classes=output_dim, average='weighted', task='multiclass')

    def forward(self, x):
        x = self.residual_layer(x)
        x = x[:, -1, :]
        x = self.final_fc(x)
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
        x, _ = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return preds

    def configure_optimizers(self):
        return self.hparams.optimizer(params=self.trainer.model.parameters())
