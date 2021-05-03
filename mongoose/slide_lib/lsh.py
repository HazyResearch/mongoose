import collections
import os
import sys
import math
import random
import numpy as np
import numpy.random
import scipy as sp
import scipy.stats

import torch
import os

from clsh import pyLSH
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class LSH:
    def __init__(self, func_, K_, L_, threads_= 8):
        self.func = func_
        self.K = K_
        self.L = L_
        self.lsh_ = pyLSH(self.K, self.L, threads_)

        self.sample_size = 0
        self.count = 0

    def setSimHash(self, func_):
        self.func = func_

    def resetLSH(self, func_):
        self.func = func_
        self.clear()

    def stats(self):
        avg_size = self.sample_size // max(self.count, 1)
        print("hashtable avg_size", avg_size)
        self.sample_size = 0
        self.count = 0
        return avg_size

    def remove_insert(self, item_id, old_item, new_fp):
        old_fp = self.func.hash(old_item).int().cpu().numpy()
        self.lsh_.remove(np.squeeze(old_fp), item_id)
        self.lsh_.insert(new_fp, item_id)

    def insert(self, item_id, item):
        fp = self.func.hash(item).int().cpu().numpy()
        self.lsh_.insert(np.squeeze(fp), item_id)

    def insert_fp(self, item_id, fp):
        self.lsh_.insert(np.squeeze(fp), item_id)

    def insert_multi(self, items, N):
        fp = self.func.hash(items).int().cpu().numpy()
        # print("lsh new: fp", fp)
        self.lsh_.insert_multi(fp, N)

    def query(self, item):
        fp = np.ascontiguousarray(self.func.hash(item).int().cpu())
        return self.lsh_.query(np.squeeze(fp))

    def query_fp(self, fp):
        fp = np.ascontiguousarray(fp)
        return self.lsh_.query(fp)

    def query_multi(self, items, N):
        #fp = np.ascontiguousarray(self.func.hash(items, transpose=True).int().cpu())
        fp = np.ascontiguousarray(self.func.hash(items, transpose=False).int().cpu())
        return self.lsh_.query_multi(fp, N), fp

    def query_multi_mask(self, item, M, N):
        fp = self.func.hash(item).int().cpu().numpy()
        #print("query_multi_mask", fp)
        mask = torch.zeros(M, N, dtype=torch.float32)
        # mask_L = torch.zeros(M, self.L, N,dtype=torch.float32)
        self.lsh_.query_multi_mask(fp, mask.numpy(), M, N)
        # self.lsh_.query_multi_mask_L(fp, mask.numpy(), mask_L.numpy(), M, N)
        return  mask.to(device), fp

    def accidental_match(self, labels, samples, N):
        self.lsh_.accidental_match(labels, samples, N)

    def multi_label(self, labels, samples):
        return self.lsh_.multi_label(labels, samples)

    def multi_label_nonunion(self, labels, mask):
        return self.lsh_.multi_label_nonunion(labels, mask)

    # def query_remove_matrix(self, items, labels, total_size):
    #     # for each data sample, query lsh data structure, remove accidental hit
    #     # find maximum number of samples
    #     # create matrix and pad appropriately
    #     batch_size, D = items.size()
    #     fp = self.func.hash(items).int().cpu().numpy()
    #     result, total_count = self.lsh_.query_matrix(fp, labels.cpu().numpy(), batch_size, total_size)
    #     batch_size, ssize = result.shape
    #     self.sample_size += total_count
    #     self.count += batch_size
    #     return result

    def query_remove_matrix(self, items, labels, total_size):
        # for each data sample, query lsh data structure, remove accidental hit
        # find maximum number of samples
        # create matrix and pad appropriately
        batch_size, D = items.size()
        fp = self.func.hash(items).int().cpu().numpy()
        result, total_count = self.lsh_.query_matrix(fp, labels, batch_size, total_size)
        batch_size, ssize = result.shape
        self.sample_size += total_count
        self.count += batch_size
        return result,fp

    def query_remove(self, item, label):
        fp = self.func.hash(item).int().cpu().numpy()
        result = self.lsh_.query(np.squeeze(fp))
        if label in result:
            result.remove(label)
        self.sample_size += len(result)
        self.count += 1
        return list(result)

    def print_stats(self):
        print("in lsh new")
        return self.lsh_.print_stats()

    def clear(self):
        self.lsh_.clear()
