import math
import torch
from torch import nn
from EncoderLayer import EncoderLayer#引入编码层内容
from Position_process import PositionalEncoding

#这里完成编码的部分
#定义编码器
class Encoder(nn.Module):
    def __init__(self,input_dim,d_model,num_heads,d_ff,num_layer,dropout):
        '''
        Transformer的编码器
        :param input_dim:输入的词表大小
        :param d_model: 词向量维度
        :param num_heads: 多头注意力的头数
        :param d_ff: FNN的隐藏层维度
        :param num_layer: Encoder的层数
        :param dropout: 正则化
        '''
        super().__init__()
        self.d_model = d_model
        self.embeddings = nn.Embedding(input_dim,d_model)#把token编号映射为词向量
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model,num_heads,d_ff,dropout) for _ in range(num_layer)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self,src,src_mask = None):
        #src:[batch_size,src_len],句子数量以及每个句子的序列长度（通常固定）
        x = self.embeddings(src) * math.sqrt(self.d_model)#这里乘一个缩放比例大小，让编码大小和位置编码规模接近，从而保持训练稳定
        x = self.pos_encoder(x)
        #随后经过多头注意力机制
        x = self.dropout(x)#正则话一次
        for layer in self.layers:
            x = layer(x,src_mask)
        return x


