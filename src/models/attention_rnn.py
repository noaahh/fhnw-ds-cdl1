import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from lightning import LightningModule
from torchmetrics import Accuracy, F1Score


class ScaledDotProductAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.scaling_factor = 1. / math.sqrt(query_dim)

    def forward(self, query, keys, values):
        # print(f"Query shape: {query.shape}")
        # print(f"Keys shape: {keys.shape}")
        # print(f"Values shape: {values.shape}")

        # keys_transposed = keys.transpose(0, 1).transpose(1, 2)
        keys_transposed = keys.transpose(1, 2)
        # print(f"Keys transposed shape: {keys_transposed.shape}")

        query = query.unsqueeze(1)
        # print(f"Query shape unsqueezed: {query.shape}")

        attention_logits = torch.bmm(query, keys_transposed)
        attention_logits_scaled = attention_logits * self.scaling_factor
        attention_weights = F.softmax(attention_logits_scaled, dim=2)
        # print(f"Attention weights shape: {attention_weights.shape}")

        output = torch.bmm(attention_weights, values)
        # print(f"Output shape: {output.shape}")

        return attention_weights.squeeze(1), output.squeeze(1)


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeatureExtractor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, x):
        outputs, _ = self.lstm(x)
        return outputs


class AttentionRnnClassifier(LightningModule):
    def __init__(self, optimizer, feature_dim, hidden_dim, num_classes=5):
        super(AttentionRnnClassifier, self).__init__()
        self.save_hyperparameters()
        self.feature_extractor = FeatureExtractor(feature_dim, hidden_dim)
        self.attention = ScaledDotProductAttention(hidden_dim * 2, hidden_dim * 2, hidden_dim * 2)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.f1_score = F1Score(num_classes=num_classes, average='weighted', task='multiclass')

    def forward(self, x):
        features = self.feature_extractor(x)  # [Batch, Sequence, Hidden * 2]
        query = features[:, -1, :]  # Use the last hidden state as query
        _, weighted_features = self.attention(query, features, features)
        logits = self.classifier(weighted_features)
        return logits

    def _shared_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        y = y.argmax(dim=1)
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
        return self.hparams.optimizer(params=self.parameters())
