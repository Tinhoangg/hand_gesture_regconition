import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (- math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head=8, dim_feedforward=512, dropout=0.3, activation_f = 'relu'):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, dropout=dropout,batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        ACTIVATION = {'relu': F.relu,
                      'gelu': F.gelu}
        self.activation = ACTIVATION[activation_f]
    def forward(self, src, src_key_padding_mask=None):
        # src: (seq_len, batch, d_model)
        attn_output, _ = self.attn(src, src, src)
        src = self.norm1(src + self.dropout(attn_output))
        ff_output = self.linear2(self.activation(self.linear1(src)))
        src = self.norm2(src + self.dropout(ff_output))
        return src
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_head=8, dim_feedforward=512, dropout=0.2,activation_f='relu', num_layers=4 ):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model=d_model, n_head=n_head, dim_feedforward=dim_feedforward, dropout=dropout, activation_f=activation_f) 
                                     for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, src,src_key_padding_mask=None):
        #src shape: (seq_len, batch, d_model)
        for layer in self.layers:
            src = layer(src)
            src = self.norm(src)
        return src
class HandGestureTransformer(nn.Module):
    def __init__(self, input_dim=126, d_model=128, n_head=8, num_layers=4, 
                 num_classes=30, dropout=0.2, max_frames=100):
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_frames)
        

        self.transformer_encoder = TransformerEncoder(num_layers=num_layers, d_model=d_model, n_head=n_head, dropout=dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, src_key_padding_mask=None):
        # x: (batch, seq_len, input_dim)
        
        # Project input
        x = self.input_projection(x)
        x = self.dropout(x)
        # Add positional encoding
        x = self.pos_encoder(x)
        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Global average pooling (considering mask)
        if src_key_padding_mask is not None:
            # Đảm bảo kiểu bool
            src_key_padding_mask = src_key_padding_mask.bool()
            
            # Đảo ngược mask để vùng hợp lệ = 1, padding = 0
            valid_mask = (~src_key_padding_mask).unsqueeze(-1).float()
            
            # Tránh chia cho 0 nếu toàn bộ là padding
            denom = valid_mask.sum(dim=1).clamp(min=1e-6)
            x = (x * valid_mask).sum(dim=1) / denom
        else:
            x = x.mean(dim=1)
        # Classification
        x = self.classifier(x)
        
        return x



    


        

