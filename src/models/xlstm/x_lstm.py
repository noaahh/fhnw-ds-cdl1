import torch
import torch.nn as nn
from lightning import LightningModule

from torch import Tensor
from torch.optim import AdamW
from torch.optim import Optimizer
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score

from typing import Any, Dict, Generator, List, Tuple, Callable, Iterable

from itertools import repeat
from einops import rearrange

from src.models.xlstm.m_lstm import mLSTM
from src.models.xlstm.s_lstm import sLSTM
from src.models.xlstm.util import Hidden

OptimizerCallable = Callable[[Iterable], Optimizer]

class xLSTM(LightningModule):
    '''The extended Long Short Term Memory (xLSTM) module as
    originally introduced in Beck et al. (2024)] see:
    (https://arxiv.org/abs/2405.04517).

    This model stacks sLSTM and mLSTM modules with residual
    connections and offers superior memory and performance
    compared to the standard LSTM model, achieving competitive
    or better performance and scaling than Transformer models
    or State-Space models.

    DISCLAIMER:
    This code was heavily inpisred by already existing implementations of the xLSTM model.

    While there wasn't one perfect one, there were 2 that were used as a base for this implementation.

    All the text embedding specific details were removed and adjusted accordingly.

    The original repositories can be found here:
    - https://github.com/muditbhargava66/PyxLSTM
    - https://github.com/myscience/x-lstm

    '''

    def __init__(
            self,
            num_layers : int,
            signature : Tuple[int, int],
            inp_dim : int,
            head_dim : int,
            head_num : int,
            output_size : int,
            p_factor : Tuple[float, float] = (2, 4/3),
            ker_size : int = 4,
            optimizer : OptimizerCallable = AdamW,
            inference_kw: Dict[str, Any] = {}
    ) -> None:
        '''Initialize the LLM model.

        Args:
            num_layers (int): The number of layers in the LLM model.
            signature (Tuple[int, int]): The signature of the LLM model,
                which represents the ration of the mLSTM-to-sLSTM blocks.
            inp_dim (int): The dimension of the input tokens.
            head_dim (int): The dimension of each attention head.
            head_num (int): The number of attention heads.
            p_factor (Tuple[float, float], optional): The expansion factor
                for the MLP projection in the m|s-LSTM blocks. Defaults to (2, 4/3).
            ker_size (int, optional): The kernel size for the causal convolutional layers.
                Defaults to 4.

            kwargs: Additional keyword arguments used at inference time (see relevant
                arguments of the generate method).
        '''
        super().__init__()

        self.accuracy = Accuracy(task='multiclass', num_classes=output_size)
        self.f1_score = F1Score(num_classes=output_size, average='weighted', task='multiclass')
        self.optimizer = optimizer
        self.inference_kw = inference_kw

        m_factor, s_factor = p_factor

        mlstm_par = {
            'inp_dim' : inp_dim,
            'head_dim' : head_dim,
            'head_num' : head_num,
            'p_factor' : m_factor,
            'ker_size' : ker_size,
        }

        slstm_par = {
            'inp_dim' : inp_dim,
            'head_dim' : head_dim,
            'head_num' : head_num,
            'p_factor' : s_factor,
            'ker_size' : ker_size,
        }

        m_num, s_num = signature
        which = [True] * m_num + [False] * s_num

        self.model : List[mLSTM | sLSTM] = nn.ModuleList([
            mLSTM(**mlstm_par) if w else sLSTM(**slstm_par)
            for w, _ in zip(repeat(which), range(num_layers))
        ])

        self.head = nn.Linear(inp_dim, output_size, bias=False)

        self.save_hyperparameters()

    def forward(
            self,
            seq: Tensor,
            hid: Hidden | None = None,
            batch_first : bool = True,
    ) -> Tuple[Tensor, Hidden]:
        '''Forward pass of the xLSTM model.

        Args:
            tok (Tensor): Input tensor representing the sequence tokens.
                Expected shape: (batch, seq_len) if batch_first=True,
                else (seq_len, batch).
            hid (Hidden, optional): Cache object for storing intermediate hidden
                values of the m|s-LSTM blocks of the model. If None, the hidden
                states are initialized by the models. Defaults to None.

        Returns:
            Tuple[Tensor, Hidden]: Returns tensor of predicted logits of shape
                (batch, seq_len, vocab_size) if batch_first=True or of shape
                (seq_len, batch, vocab_size) if batch_first=False, and the
                updated hidden model states.
        '''


        if batch_first: seq = rearrange(seq, 'b s i -> s b i')
        if hid is None: hid = [l.init_hidden(seq.size(1)) for l in self.model]

        # Pass the sequence through the mLSTM and sLSTM blocks
        out = []
        for inp in seq:
            # Compute model output and update the hidden states
            for i, lstm in enumerate(self.model):
                inp, hid[i] = lstm(inp, hid[i])

            out.append(inp)

        out = torch.stack(out, dim=1 if batch_first else 0)
        out = self.head(out)
        out = out[:, -1, :]

        return out, hid

    def _shared_step(self, batch, batch_idx):
        x, y = batch
        logits, hid = self(x)
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

    def configure_optimizers(self) -> Optimizer:
        optim = self.optimizer(
            self.parameters(),
        )

        return optim
