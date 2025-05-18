from audioop import cross

import torch
from torch import nn
from torch.nn import functional as F
from FNN import FeedForward
from Multi_HeadAttention import MultiHeadAttention

#这里主要做解码层的输入输出管理，主要是三层，首先是带掩蔽的多头交叉注意力机制，其次是结合Encoder输入的自注意力机制，随后仍然为FFN
class DecoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model,num_heads)
        self.cross_attn = MultiHeadAttention(d_model,num_heads)
        self.fnn = FeedForward(d_model,d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,enc_output,src_mask=None,trg_mask =None):
        #首先过的是self_attn，而且带掩蔽
        self_attn_output = self.self_attn(x,x,x,trg_mask)
        x = x + self.dropout(self_attn_output)
        x = self.norm1(x)
        #接着到了cross_attention
        cross_output = self.cross_attn(x,enc_output,enc_output,src_mask)#注意输入是Q,K,V形式
        #紧接着残差
        x = x + self.dropout(cross_output)
        x = self.norm2(x)
        #紧接着再过FFN
        x = x + self.dropout(self.fnn(x))
        x = self.norm3(x)
        return x

