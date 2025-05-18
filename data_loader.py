import torch
from torch.utils.data import Dataset

#创建数据集和数据加载器
class TranslationDataset(Dataset):
    def __init__(self,src_sequences,trg_sequences):
        self.src_sequences = src_sequences
        self.trg_sequences = trg_sequences

    def __len__(self):
        return len(self.src_sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.src_sequences[idx]),torch.tensor(self.trg_sequences[idx])

