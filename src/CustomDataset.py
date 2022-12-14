import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Literal, Dict, List
import random, re, os
import numpy as np

# For Reorganization Energy Regression (ground state + excite state)
class SMILESDataset(Dataset):
    def __init__(self, tokenizer : Dict, df : pd.DataFrame, max_len : int = 128, mode : Literal['train', 'submission'] = 'train', cols : List = ["Reorg_g", "Reorg_ex"]):
        self.max_len = max_len
        self.mode = mode
        self.df = df
        self.tokenizer = tokenizer
        self.cols = cols

        if self.mode == "submission":
            self.label_array = None
        else:
            self.label_array = df[cols].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx : int):
        item = self.df.iloc[idx]
        item = self._process(item)
        return item

    def _process(self, row):
        seq = row['SMILES']

        if self.mode != 'submission':
            target = row[self.cols]
            target = torch.Tensor(target)
        else:
            target = None

        if len(seq) > self.max_len:
            idx = random.randint(0,len(seq)-self.max_len)
            seq = seq[idx:idx+self.max_len]

        # seq = re.sub(r"[UZOB]", "X", seq)
        tokens = list(seq)
        tokens += ['<PAD>' for _ in range(self.max_len - len(tokens))]

        ret = np.array([self.tokenizer[v] if v in list(self.tokenizer.keys()) else len(list(self.tokenizer.keys())) for v in tokens ])
        ret = torch.from_numpy(ret)

        if self.mode == 'train':
            data = {}
            data['seq'] = ret
            data['y'] = target
            return data
        else:
            data = {}
            data['seq'] = ret
            data['y'] = None
            return data
