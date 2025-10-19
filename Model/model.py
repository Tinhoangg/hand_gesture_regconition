from keypoint_detect import Keypoint_detect
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import torch.optim as optim
import math

class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (- math.log(1000) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_forward = 2048, dropout=0.2):
        '''
        d_model: kích thước vector đầu vào feed forward
        n_head: số head trong multi-head attention
        dim_forward: số node trong layer
        dropout: tỉ lệ dropout'''
        super.__init__()

        #Multi-head attention
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, dropout=dropout)

        # Feed forward network
        self.linear1 = nn.Linear(in_features= d_model, out_features=dim_forward)
        self.linear2 = nn.linear(dim_forward, d_model)

        #  layer norm va dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        #src shape: (seq_len, batch_size, d_model)

        attn_output, _ = self.self_attn(src, src, src)

        # residual conneection + normalization
        src = self.norm1(src + self.dropout(attn_output))

        # feed forward network
        ff_output = self.linear2(F.relu(self.linear1(src)))
        src = self.norm2(src + self.dropout(ff_output))

        return src

        

