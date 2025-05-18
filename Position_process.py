import math
import torch
from torch import nn

#这里主要实现位置编码的部分
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):  # 统一这里
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # shape: [5000, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self,x):#传入的是一组embedding好的序列
        '''
        :param x: [batch_size,seq_len,d_model]
        :return: word_embedding + positional_embedding
        '''
        x += self.pe[:,:x.size(1)].to(x.device)
        return x