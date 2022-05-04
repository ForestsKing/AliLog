import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class FullAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(FullAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads

        self.W_Q = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.d_v, self.d_model, bias=False)

        self.norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_Q, input_K, input_V, attn_mask=None):
        residual, batch_size = input_Q.clone(), input_Q.size(0)

        Q = (self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2))
        K = (self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2))
        V = (self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2))

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)

        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2)
        context = context.reshape(batch_size, -1, self.n_heads * self.d_v)

        output = self.dropout(self.fc(context))
        return self.norm(output + residual)


class AttenLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, dropout):
        super(AttenLayer, self).__init__()
        self.attention = FullAttention(d_k, d_v, d_model, n_heads, dropout)
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        x = self.attention(q, k, v, attn_mask=attn_mask)

        residual = x.clone()
        y = self.dropout(self.fc2(self.fc1(x)))
        return self.norm(residual + y)
