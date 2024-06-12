# path/filename: /path/to/your_pytorch_lightning_wrapper.py
import torch
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy, F1Score
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)
import torch.nn.functional as F

class XLSTM(L.LightningModule):
    def __init__(self, config, optimizer):
        super().__init__()
        self.save_hyperparameters()

        self.model = xLSTMBlockStack(
            xLSTMBlockStackConfig(
                mlstm_block=mLSTMBlockConfig(
                    mlstm=mLSTMLayerConfig(
                        conv1d_kernel_size=config['mlstm_conv1d_kernel_size'],
                        qkv_proj_blocksize=config['mlstm_qkv_proj_blocksize'],
                        num_heads=config['mlstm_num_heads']
                    )
                ),
                slstm_block=sLSTMBlockConfig(
                    slstm=sLSTMLayerConfig(
                        backend=config['slstm_backend'],
                        num_heads=config['slstm_num_heads'],
                        conv1d_kernel_size=config['slstm_conv1d_kernel_size']
                    ),
                    feedforward=FeedForwardConfig(
                        proj_factor=config['slstm_proj_factor'],
                        act_fn=config['slstm_act_fn']
                    )
                ),
                context_length=config['context_length'],
                num_blocks=config['num_blocks'],
                embedding_dim=config['embedding_dim'],
                slstm_at=config['slstm_at'],
            )
        )
        self.classifier = nn.Linear(config['embedding_dim'], config['num_classes'])
        self.accuracy = Accuracy(task='multiclass', num_classes=config['num_classes'])
        self.f1_score = F1Score(num_classes=config['num_classes'], average='weighted', task='multiclass')

    def forward(self, x):
        features = self.model(x)
        output = self.classifier(features[:, -1, :])
        return output

    def _shared_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1).float()
        loss = F.cross_entropy(logits, y.float())
        y = torch.argmax(y, dim=1).float()
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
        return loss

    def configure_optimizers(self):
        return self.hparams.optimizer(params=self.trainer.model.parameters())
