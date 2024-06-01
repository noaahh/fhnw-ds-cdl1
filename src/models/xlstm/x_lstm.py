import torch
import torch.nn as nn

from s_lstm import sLSTM
from m_lstm import mLSTM
import torch.nn.functional as F
from lightning import LightningModule

class xLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_blocks,
                 dropout=0.0, bidirectional=False, lstm_type="slstm"):
        super(xLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.lstm_type = lstm_type

        # clean cycle for the blocks so input_size -> hidden_size -> input_size -> contrary to input_size -> hidden_size -> hidden_size -> output_size
        self.blocks = nn.ModuleList([xLSTMBlock(input_size, hidden_size, num_layers, dropout, bidirectional, lstm_type)
                                     for i in range(num_blocks)])

        # self.blocks = nn.ModuleList([xLSTMBlock(input_size if i==0 else hidden_size, hidden_size, num_layers, dropout, bidirectional, lstm_type)
        #                              for i in range(num_blocks)])

        self.output_layer = nn.Linear(input_size, output_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, input_seq, hidden_states=None):
        if hidden_states is None:
            hidden_states = [None] * self.num_blocks

        output_seq = input_seq
        for i, block in enumerate(self.blocks):
            output_seq, hidden_state = block(output_seq, hidden_states[i])
            if self.lstm_type == "slstm":
                hidden_states[i] = [[hidden_state[j][0].detach(), hidden_state[j][1].detach()] for j in range(len(hidden_state))]
            else:
                hidden_states[i] = hidden_state

        output_seq = self.output_layer(output_seq)
        return output_seq, hidden_states

class xLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0, bidirectional=False, lstm_type="slstm"):
        super(xLSTMBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.lstm_type = lstm_type

        if lstm_type == "slstm":
            self.lstm = sLSTM(input_size, hidden_size, num_layers, dropout)
        elif lstm_type == "mlstm":
            print("Warning: mLSTM is not working yet.")
            self.lstm = mLSTM(input_size, hidden_size, num_layers, dropout)
        else:
            raise ValueError(f"Invalid LSTM type: {lstm_type}")

        self.norm = nn.LayerNorm(input_size)
        self.activation = nn.GELU()
        self.dropout_layer = nn.Dropout(dropout)

        if bidirectional:
            self.proj = nn.Linear(2 * hidden_size, input_size)
        else:
            self.proj = nn.Linear(hidden_size, input_size)

        # print shapes
        print(f"input_size: {input_size}")
        print(f"hidden_size: {hidden_size}")
        print(f"num_layers: {num_layers}")
        print(f"dropout: {dropout}")
        print(f"proj: {self.proj}")

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, input_seq, hidden_state=None):
        lstm_output, hidden_state = self.lstm(input_seq, hidden_state)
        if self.lstm_type == "slstm":
            hidden_state = [[hidden_state[i][0].detach(), hidden_state[i][1].detach()] for i in range(len(hidden_state))]

        if self.bidirectional:
            lstm_output = torch.cat((lstm_output[:, :, :self.hidden_size], lstm_output[:, :, self.hidden_size:]), dim=-1)

        output = self.activation(self.proj(lstm_output))
        output = self.norm(output + input_seq)
        output = self.dropout_layer(output)

        return output, hidden_state

# Example of how to initialize the xLSTMLightning module
if __name__ == "__main__":
    model = xLSTM(input_size=16, hidden_size=256, output_size=10, num_layers=3, num_blocks=6, lstm_type="slstm")
    input_seq = torch.randn(10, 32, 16)  # Batch size = 10, sequence length = 32, feature size = 16
    output_seq = model(input_seq)
    print(output_seq.shape)  # Output shape should be [batch size, sequence length, output size]
