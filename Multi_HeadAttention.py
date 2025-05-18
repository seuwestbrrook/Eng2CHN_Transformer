import torch
from torch import nn


#定义多头注意力机制
#首先定义的是多头注意力机制的计算模块
class ScaleDotProductAttention(nn.Module):#核心计算公式模块
    def __init__(self,d_k):
        super(ScaleDotProductAttention,self).__init__()#启动
        self.scale = d_k ** -0.5#缩放因子

    def forward(self,Q,K,V,mask = None):#默认不用mask
        '''
        进行注意力机制的公式计算
        :param Q: [batch_size,heads,seq_len,d_k]
        :param K: [batch_size,heads,seq_len,d_k]
        :param V: [batch_size,heads,seq_len,d_k]
        :param mask: [batch_size,1,1,seq_len]
        :return: output and weight
        '''
        scores = torch.matmul(Q,K.transpose(-2,-1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0,float('-inf'))#将序列中位0的位置设置为负无穷，表示不能访问
        attn = torch.softmax(scores,dim=-1)
        output = torch.matmul(attn,V)
        return output,attn

#接下来是多头注意力机制的神经网络，主要作用是进行K,Q,V的转换
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_k = self.d_v = d_model // num_heads
        self.num_heads = num_heads

        # 定义线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        self.attention = ScaleDotProductAttention(self.d_k)
        self.dropout = nn.Dropout(0.1)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 线性变换并分头
        Q = self.w_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        K = self.w_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        V = self.w_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)

        # 计算注意力
        if mask is not None:
            #mask = mask.unsqueeze(1)  # 扩展维度以匹配多头这里是不需要匹配的，因为输入已经加了一层了，这里再扩展维度就对不上了
            pass
        output, attn = self.attention(Q, K, V, mask=mask)

        # 拼接多头的输出
        output = output.transpose(1,2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.fc(output)
        return output
