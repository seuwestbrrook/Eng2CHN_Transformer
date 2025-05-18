import math
import torch
from torch import nn
from Position_process import PositionalEncoding
from DecoderLayer import DecoderLayer

class Decoder(nn.Module):
    def __init__(self,output_dim,d_model,num_heads,d_ff,num_layers,dropout):
        super().__init__()
        self.d_model = d_model
        self.embeddings = nn.Embedding(output_dim,d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model,num_heads,d_ff,dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model,output_dim)#这里是Decoder的输出层,输出的output_dim会用来做分类任务

    def forward(self,trg,enc_output,src_mask = None,trg_mask = None):
        #trg:[batch_size,trg_len]这里主要是处理意见输出的内容
        x = self.embeddings(trg) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)#加上位置信息
        x = self.dropout(x)#过一层正则层
        for layer in self.layers:
            x = layer(x,enc_output,src_mask,trg_mask)
        output = self.fc_out(x)
        return output