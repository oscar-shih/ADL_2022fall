import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
import random



class MCDataset(Dataset):
    def __init__(self, args, tokenizer) -> None:
        super(MCDataset, self).__init__()
        self.data = []
        pass
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]
    def collate_fn(self):
        pass


class QADataset(Dataset):
    def __init__(self, args, tokenizer) -> None:
        super(QADataset, self).__init__()
        self.data = []
        pass
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]
    def collate_fn(self):
        pass

