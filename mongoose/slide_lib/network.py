import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from mongoose.slide_lib.simHash import SimHash
from mongoose.slide_lib.lsh import LSH


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

np.random.seed(1234)
torch.manual_seed(1234)


class LSHSampledLayer(nn.Module):
    def __init__(self, hash_weight, layer_size, K, L, num_class):
        super(LSHSampledLayer, self).__init__()
        self.D = layer_size
        self.K = K
        self.L = L
        self.num_class = num_class
        self.hash_weight = hash_weight

        self.store_query = True
        # last layer
        self.params = nn.Linear(layer_size, num_class)
        self.params.bias = nn.Parameter(torch.Tensor(num_class, 1))
        self.init_weights(self.params.weight, self.params.bias)

        # construct lsh using triplet weight
        self.lsh = None
        self.initializeLSH()

        self.count = 0
        self.sample_size = 0

        self.thresh_hash = SimHash(self.D+1, 1, self.L)
        self.thresh = 0.3
        self.hashcodes = self.thresh_hash.hash(torch.cat((self.params.weight, self.params.bias), dim = 1))

        for name in self.backward_timer_names:
            self.backward_timer[name] = []
        for name in self.forward_timer_names:
            self.forward_timer[name] = []


    def initializeLSH(self):
        self.lsh = LSH( SimHash(self.D+1, self.K, self.L, self.hash_weight), self.K, self.L )
        weight_tolsh = torch.cat( (self.params.weight, self.params.bias), dim = 1)
        self.lsh.insert_multi(weight_tolsh.to(device).data, self.num_class )
    
    def setSimHash(self, seed, hashweight = None):
        print("update simhash")
        if(hashweight!=None):
            self.lsh.setSimHash( SimHash(self.D+1, self.K, self.L, hashweight ) )

    def rebuild(self):
        weight_tolsh = torch.cat((self.params.weight, self.params.bias), dim=1)
        check = self.thresh_hash.hash(weight_tolsh)
        distance = check - self.hashcodes
        if torch.sum(torch.abs(distance))>self.thresh*distance.numel():
            print("Rebuild LSH")
            self.lsh.clear()
            #include bias
            self.lsh.insert_multi(weight_tolsh.to(device).data, self.num_class )
            self.hashcodes = check
        else:
            print("No need")

    def init_weights(self, weight, bias):
        initrange = 0.05
        weight.data.uniform_(-initrange, initrange)
        bias.data.fill_(0)
        # bias.require_gradient = False

    def train_forward(self, x, y, triplet_flag, debug=False):
        '''weight normalization'''

        N, D = x.size()
        # sid, hashcode = self.lsh.query_multi(x.data, N)
        query_tolsh = torch.cat( (x, torch.ones(N).unsqueeze(dim = 1).to(device)), dim = 1 )
        sid, hashcode = self.lsh.query_multi(query_tolsh.data, N)

        sampled_ip = 0
        sampled_cos = 0
        sizes = 0
        retrieved_size = sizes * 1.0 / N

        # add y
        sid_list, target_matrix = self.lsh.multi_label(y.data.cpu().numpy(), sid)
        new_targets = Variable(torch.from_numpy(target_matrix)).to(device)

        sample_ids = Variable(torch.from_numpy(np.asarray(sid_list, dtype=np.int64)), requires_grad=False).to(device)
        sample_size = sample_ids.size(0)
        # print("sample size", sample_size)
        sample_weights = F.embedding(sample_ids, self.params.weight, sparse=True)
        sample_bias = self.params.bias.squeeze()[sample_ids]
        sample_product = sample_weights.matmul(x.t()).t()
        sample_logits = sample_product + sample_bias
        self.lsh.sample_size += sample_size
        weight_pair = {}
        return sample_logits, new_targets, sample_size, retrieved_size, weight_pair, hashcode, sampled_ip, sampled_cos

    def forward(self, x, y, triplet_flag, debug):
        if self.training:
            # self.speed_test(x, y)
            return self.train_forward(x, y, triplet_flag, debug)
        else:
            return torch.matmul(x, self.params.weight.t()) + self.params.bias.squeeze()


class Net(nn.Module):
    def __init__(self, input_size, output_size, layer_size, hash_weight, K, L):
        super(Net, self).__init__()
        stdv = 1. / math.sqrt(input_size)
        self.input_size = input_size
        self.output_size = output_size
        self.layer_size = layer_size
        self.fc = nn.Embedding(self.input_size + 1, 128, padding_idx=input_size, sparse=True)
        self.bias = nn.Parameter(torch.Tensor(layer_size))
        self.bias.data.uniform_(-stdv, stdv)
        self.lshLayer = LSHSampledLayer(hash_weight, layer_size, K, L, output_size)

    def forward(self, x, y, triplet_flag, debug):
        emb = torch.sum(self.fc(x), dim=1)
        emb = emb / torch.norm(emb, dim=1, keepdim=True)
        query = F.relu(emb + self.bias)
        return self.lshLayer.forward(query, y, triplet_flag, debug)


    def forward_full(self, x, y, t1, t2):
        with torch.no_grad():
            N,d = x.size()
            emb = torch.sum(self.fc(x), dim=1)
            emb = emb / torch.norm(emb, dim=1, keepdim=True)
            query = F.relu(emb + self.bias)
            product = torch.matmul(query, self.lshLayer.params.weight.t())

            t1 = 0.001
            t2 = 0.5
            t1_th = int(self.output_size * (1 - t1) )  # positive inner product rank
            t2_th = int(self.output_size * (1 - t2) )  # negative inner product rank
            t1_ip = torch.mean( torch.kthvalue(product, t1_th )[0]).item()
            t1_ip = max(0.0, t1_ip)
            t2_ip = torch.mean( torch.kthvalue(product, t2_th )[0]).item()

            #ip threshold mask
            ip_t1_mask = product > t1_ip
            ip_t2_mask = product < t2_ip

            query = torch.cat( (query.data, torch.ones(N).unsqueeze(dim = 1).to(device)), dim = 1 )

            retrieved, _ = self.lshLayer.lsh.query_multi_mask(query, N, self.output_size)
            retrieved = retrieved.bool()

            positive_mask = ip_t1_mask & (~retrieved)
            negative_mask = ip_t2_mask & retrieved

            num_negative = torch.sum(negative_mask).item()
            num_positive = torch.sum(positive_mask).item()

            row, column = torch.where(positive_mask == 1)
            p_arc = query[row].detach()
            p_pos = torch.cat( (self.lshLayer.params.weight[column], self.lshLayer.params.bias[column]), dim = 1).detach()


            assert p_arc.size()[0]==p_pos.size()[0]
            assert p_arc.size()[1]==p_pos.size()[1]

            row, column = torch.where(negative_mask == 1)
            n_arc = query[row].detach()

            n_neg = torch.cat( (self.lshLayer.params.weight[column], self.lshLayer.params.bias[column]), dim = 1).detach()

            assert n_arc.size()[0]==n_neg.size()[0]
            assert n_arc.size()[1]==n_neg.size()[1]

            #down sample 
            size = min(num_negative, num_positive)
            if(num_positive < num_negative):
                random_perm = torch.randperm(num_negative)
                permute_id=random_perm[: int(num_positive)]
                n_arc = n_arc[permute_id]
                n_neg = n_neg[permute_id]
                num_negative = n_arc.size()[0]
            else:
                random_perm = torch.randperm(num_positive)
                permute_id=random_perm[:int(num_negative)]
                p_arc = p_arc[permute_id]
                p_pos = p_pos[permute_id]
                num_positive = p_arc.size()[0]

        return p_arc,p_pos,n_arc,n_neg,num_negative


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
                # fvs = list()
                for fdx in range(1, len(items), 1):
                    fid, fv = items[fdx].split(":")
                    ids.append(int(fid))
                    # fvs.append( float(fv) )
                self.max_D = max(self.max_D, len(ids))
                self.data.append([torch.from_numpy(np.asarray(x)) for x in [labels, ids]])

                # if idx % 100000 == 0:
                #     print(idx)

    def pad(self, item, width, value):
        result = torch.zeros(width).long()
        result.fill_(value)
        result[:len(item)] = item
        return result

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        labels, idxs = self.data[idx]
        return self.pad(labels, self.max_L, -1), self.pad(idxs, self.max_D, self.D)
