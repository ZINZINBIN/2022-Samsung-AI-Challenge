import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from typing import List, Optional, Union

class SelfAttention(nn.Module):
    def __init__(
        self, 
        batch_size : int, 
        vocab_size : int, 
        dropout : float = 0.5, 
        embedd_config = None
        ):

        super(SelfAttention, self).__init__()
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.embedd_hidden_size = embedd_config["hidden_dim_l0"]
        self.embedd_dims = embedd_config["embedd_dim"]

        self.embeddings = nn.Embedding(vocab_size, self.embedd_dims)

        # antigen path
        self.lstm = nn.LSTM(self.embedd_dims, embedd_config["hidden_dim_l0"], dropout=self.dropout, bidirectional=True)
        self.w_s1 = nn.Linear(2*embedd_config["hidden_dim_l0"], embedd_config["hidden_dim_l1"])
        self.w_s2 = nn.Linear(embedd_config["hidden_dim_l1"], embedd_config["hidden_dim_l2"])

        linear_input_dims = embedd_config["hidden_dim_l0"]
        linear_input_dims = 2 * linear_input_dims

        self.regressor = nn.Sequential(
            nn.Linear(linear_input_dims, linear_input_dims // 2),
            nn.BatchNorm1d(linear_input_dims // 2),
            nn.GELU(),
            nn.Linear(linear_input_dims // 2, linear_input_dims // 2),
            nn.BatchNorm1d(linear_input_dims // 2),
            nn.GELU(),
            nn.Linear(linear_input_dims // 2, 2)
        )

    def attention(self, lstm_output:torch.Tensor)->torch.Tensor:
        attn_weight_matrix = self.w_s2(torch.tanh(self.w_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)
        return attn_weight_matrix
    
    def forward(self, seq:torch.Tensor, batch_size=None)->torch.Tensor:
        x = self.embeddings(seq)
        x = x.permute(1, 0, 2)

        if batch_size is None:
            h_0 = Variable(torch.zeros(2, self.batch_size, self.embedd_hidden_size).cuda())
            c_0 = Variable(torch.zeros(2, self.batch_size, self.embedd_hidden_size).cuda())

        else:
            h_0 = Variable(torch.zeros(2, batch_size, self.embedd_hidden_size).cuda())
            c_0 = Variable(torch.zeros(2, batch_size, self.embedd_hidden_size).cuda())

        # attention
        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        output = output.permute(1, 0, 2)
        att = self.attention(output)
        hidden = torch.bmm(att, output)

        # regression
        hidden = hidden.mean(dim = 1).view(hidden.size()[0], -1)
        y = self.regressor(hidden)

        return y