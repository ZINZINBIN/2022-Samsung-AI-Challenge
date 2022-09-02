import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(1,0,2)
        x = x + self.pe[:x.size(0), :]
        x = x.permute(1,0,2)
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(
        self, 
        seq_len : int,
        ntoken: int, 
        d_model: int, 
        nhead: int, 
        d_hid: int, 
        nlayers: int, 
        dropout: float = 0.5, 
        ):
        super().__init__()
        self.pos = PositionalEncoding(d_model, dropout, seq_len)
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(d_model, nhead, d_hid, dropout),
            nlayers
        )

        self.enc = nn.Embedding(ntoken, d_model)
        self.d_model = d_model

        linear_input_dims = d_model * seq_len

        self.regressor = nn.Sequential(
            nn.Linear(linear_input_dims, linear_input_dims // 2),
            nn.BatchNorm1d(linear_input_dims // 2),
            nn.GELU(),
            nn.Linear(linear_input_dims // 2, linear_input_dims // 2),
            nn.BatchNorm1d(linear_input_dims // 2),
            nn.GELU(),
            nn.Linear(linear_input_dims // 2, 2)
        )

        self.src_mask = None
    
    def _generate_square_subsequent_mask(self, size : int)->torch.Tensor:
        mask = (torch.triu(torch.ones(size,size))==1).transpose(0,1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, data)->torch.Tensor:

        seq = data['seq'].cuda()
        batch_size = len(seq)

        if self.src_mask is None or self.src_mask.size(0) != len(seq):
            device = seq.device
            mask = self._generate_square_subsequent_mask(len(seq)).to(device)
            self.src_mask = mask
        
        x = self.enc(seq)
        x = self.pos(x)
        x = self.transformer(x, self.src_mask) # (seq_len, batch, feature_dims)
        x = x.permute(1,0,2).contiguous().view(x.size()[0], -1)
        output = self.regressor(x)
        return output