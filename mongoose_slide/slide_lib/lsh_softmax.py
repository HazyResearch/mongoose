import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from mongoose_slide.slide_lib.simHash import SimHash
from mongoose_slide.slide_lib.lsh import LSH
import time

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class LSHSoftmax(nn.Module):
    def __init__(self, N, D, K, L, freq):
        super(LSHSoftmax, self).__init__()
        self.D = D
        self.N = N
        self.K = K
        self.L = L

        # Rebuild Settings
        self.freq = freq
        self.count = 0
        self.sample_size = 0
        self.lsh = LSH(SimHash(D, K, L), K, L)

        self.params = nn.Linear(D, N)
        self.init_weights(self.params.weight, self.params.bias)

    def init_weights(self, weight, bias):
        initrange = 0.05
        weight.data.uniform_(-initrange, initrange)
        bias.data.fill_(0)

    def build(self, lsh):
        # lsh.stats()
        lsh.clear()
        lsh.insert_multi(self.params.weight.to(device).data, self.N)

    def sampled(self, inputs, labels, debug=False):
        if self.lsh.count % self.freq == 0:
            print("RESET HASH!!!")
            self.build(self.lsh)
            # self.lsh.stats()

        # Query LSH Database
        t1 = time.time()
        N, D = inputs.size()
        # sid = self.lsh.query_multi(inputs.data, N)
        sid, hashcode = self.lsh.query_multi(inputs.data, N)
        # print("sid length",len(sid))
        # print("sid",sid)
        sampled_ip = 0
        sampled_cos = 0

        if debug:
            product_list = []
            dif_list = []
            cos_list = []
            for idex, h in enumerate(hashcode):
                d = inputs[idex]
                sid_debug = self.lsh.query_fp(h)
                if len(sid_debug) > 0:
                    retrieved = Variable(torch.from_numpy(np.asarray(list(sid_debug))), requires_grad=False).to(
                        device).long()

                    random_sample_ids = torch.randint(0, self.N, (int(len(sid_debug)),)).to(device)
                    random_weights = F.embedding(random_sample_ids, self.params.weight, sparse=True)
                    random_product = random_weights.matmul(d.t()).t()
                    random_cos = (1 - torch.acos(
                        F.cosine_similarity(d.repeat(1, random_weights.size()[0]).view(random_weights.size()[0], -1),
                                            random_weights)) / 3.141592653)
                    weights = F.embedding(retrieved, self.params.weight, sparse=True)
                    product = weights.matmul(d.t()).t()
                    cos = (1 - torch.acos(
                        F.cosine_similarity(d.repeat(1, weights.size()[0]).view(weights.size()[0], -1),
                                            weights)) / 3.141592653)
                    cos_list += [torch.mean(cos).item() - torch.mean(random_cos).item()]
                    product_list += [torch.mean(product).item()]
                    dif_list += [torch.mean(product).item() - torch.mean(random_product).item()]
            # print("mean of product_list", torch.mean( torch.from_numpy( np.asarray( product_list))))
            print("mean of sampe - random ", torch.mean(torch.from_numpy(np.asarray(dif_list))))
            print("mean of sampe - random cos", torch.mean(torch.from_numpy(np.asarray(cos_list))))
            # print("retrieved size: ", len(sid))
            sampled_ip = torch.mean(torch.from_numpy(np.asarray(dif_list)))
            sampled_cos = torch.mean(torch.from_numpy(np.asarray(cos_list)))

        sid_list, target_matrix = self.lsh.multi_label(labels.data.cpu().numpy(), sid)
        new_targets = Variable(torch.from_numpy(target_matrix)).to(device)

        sample_ids = Variable(torch.from_numpy(np.asarray(sid_list, dtype=np.int64)), requires_grad=False).to(device)
        sample_size = sample_ids.size(0)
        self.lsh.sample_size += sample_size
        self.lsh.count += 1
        # print(sample_size)

        # gather sample ids - weights and frequencies
        t1 = time.time()
        sample_weights = F.embedding(sample_ids, self.params.weight, sparse=True)
        sample_bias = self.params.bias[sample_ids]
        sample_logits = sample_weights.matmul(inputs.t()).t() + sample_bias

        # self.lsh.stats()

        return sample_logits, new_targets, sample_size, sampled_ip, sampled_cos

    def forward(self, inputs, labels, debug=False):
        if self.training:
            return self.sampled(inputs, labels, debug)
        else:
            logits = torch.matmul(inputs, self.params.weight.t()) + self.params.bias
            return logits
