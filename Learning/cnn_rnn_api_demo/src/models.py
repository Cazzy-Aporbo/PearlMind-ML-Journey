
# src/models.py
import math
from typing import Optional
import torch
import torch.nn as nn

class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, emb_dim=256, hidden_dim=512, num_layers=2, dropout=0.2, tie_weights=True):
        super().__init__()
        self.encoder = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.drop = nn.Dropout(dropout)
        if tie_weights and hidden_dim == emb_dim:
            self.decoder.weight = self.encoder.weight
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.encoder.weight, -0.1, 0.1)
        nn.init.zeros_(self.decoder.weight)

    def forward(self, x, h: Optional[tuple]=None):
        emb = self.drop(self.encoder(x))
        out, h = self.rnn(emb, h)
        out = self.drop(out)
        logits = self.decoder(out)
        return logits, h

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        pad = (kernel_size - 1) * dilation
        super().__init__(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation)
        self._causal_pad = pad

    def forward(self, x):
        out = super().forward(x)
        if self._causal_pad > 0:
            out = out[:, :, :-self._causal_pad]
        return out

class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, dilation=1, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.ln1 = nn.LayerNorm(channels)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.ln2 = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, T, C) -> conv expects (B, C, T)
        residual = x
        x = x.transpose(1, 2)
        x = self.conv1(x).transpose(1, 2)
        x = self.ln1(x)
        x = self.act(x)
        x = self.drop(x)
        x2 = x.transpose(1, 2)
        x2 = self.conv2(x2).transpose(1, 2)
        x2 = self.ln2(x2)
        x = self.drop(self.act(x2)) + residual
        return x

class TCNLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, emb_dim=256, channels=384, n_blocks=6, kernel_size=5, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.proj = nn.Linear(emb_dim, channels)
        blocks = []
        for i in range(n_blocks):
            blocks.append(TCNBlock(channels, kernel_size=kernel_size, dilation=2**i, dropout=dropout))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(channels)
        self.head = nn.Linear(channels, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x = self.proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        logits = self.head(x)
        return logits
