import torch
from torch import nn

from Multi_HeadAttention import MultiHeadAttention
from FNN import FeedForward

#这里主要定义编码层，编码层有两层



class EncoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model,num_heads)#构建多头注意力层
        self.ffn = FeedForward(d_model,d_ff,dropout)
        self.norm1 = nn.LayerNorm(d_model)#层归一化
        self.norm2 = nn.LayerNorm(d_model)#第二层层归一化
        self.dropout = nn.Dropout(dropout)#加入正则化

    def forward(self,x,mask=None):
        '''
        进行Encoder的输入到输出处理
        :param x: word_embedding + position_embedding
        :param mask: 注意力是否考虑掩蔽
        :return: 输出到Decoder多头注意力层的V与K
        '''
        attn_output = self.self_attn(x,x,x,mask = None)
        #输出后需要做残差连接
        x = x + self.dropout(attn_output)#加上正则化后再做残差连接
        x = self.norm1(x)#进行一层归一化
        x = x + self.dropout(self.ffn(x))#直接过ffn
        x = self.norm2(x)
        return x
