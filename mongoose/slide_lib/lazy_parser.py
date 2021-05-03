import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MultiLabelDataset(Dataset):
    def __init__(self, filename):
        self.build(filename)

    def build(self, filename):
        with open(filename) as f:
            metadata = f.readline().split()
            self.N = int(metadata[0])
            self.D = int(metadata[1])
            self.L = int(metadata[2])
            self.max_L = 0
            self.max_D = 0
            
            self.data = list()
            for idx in range(self.N):
                items = f.readline().split()
                labels = [int(x) for x in items[0].split(",")]
                self.max_L = max(self.max_L, len(labels))
                
                ids = list()
                for fdx in range(1, len(items), 1):
                    fid, fv = items[fdx].split(":")
                    ids.append( int(fid) )
                self.max_D = max(self.max_D, len(ids))
                self.data.append( [torch.from_numpy(np.asarray(x)) for x in [labels, ids]] )

                if idx % 100000 == 0:
                    print(idx)

    def pad(self, item, width, value):
        result = torch.zeros(width).long()
        result.fill_(value)
        result[:len(item)] = item
        return result

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        labels, data = self.data[idx]
        return self.pad(labels, self.max_L, -1), self.pad(data, self.max_D, self.D)

class ValidDataset(Dataset):
    def __init__(self, filename):
        self.build(filename)

    def build(self, filename):
        with open(filename) as f:
            metadata = f.readline().split()
            self.N = int(int(metadata[0]) * 0.025)
            #self.N = int(metadata[0])
            self.D = int(metadata[1])
            self.L = int(metadata[2])
            self.max_L = 0
            self.max_D = 0
            
            self.data = list()

            for idx in range(self.N):
                items = f.readline().split()
                labels = [int(x) for x in items[0].split(",")]
                self.max_L = max(self.max_L, len(labels))
                
                ids = list()
                for fdx in range(1, len(items), 1):
                    fid, fv = items[fdx].split(":")
                    ids.append( int(fid) )
                self.max_D = max(self.max_D, len(ids))
                self.data.append( [torch.from_numpy(np.asarray(x)) for x in [labels, ids]] )

                if idx % 100000 == 0:
                    print(idx)

    def pad(self, item, width, value):
        result = torch.zeros(width).long()
        result.fill_(value)
        result[:len(item)] = item
        return result

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        labels, data = self.data[idx]
        return self.pad(labels, self.max_L, -1), self.pad(data, self.max_D, self.D)
