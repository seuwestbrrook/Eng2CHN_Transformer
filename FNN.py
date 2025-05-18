import torch
from torch import nn
from torch.nn import functional as F
#定义FNN这个前馈神经网络
class FeedForward(nn.Module):
    def __init__(self,d_model,dff,dropout = 0.1):#dff是自定义隐藏层维度
        super().__init__()
        self.fc1 = nn.Linear(d_model,dff)
        self.fc2 = nn.Linear(dff,d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)#激活
        x = self.dropout(x)#正则化
        x = self.fc2(x)
        return x

