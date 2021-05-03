import math
import torch
import torch.nn as nn
from torch.nn import Identity
import torch.nn.functional as F
from torch.autograd import Function
from functools import partial, reduce, wraps
from itertools import chain
from operator import mul

from local_attention import LocalAttention
from axial_positional_embedding import AxialPositionalEmbedding
from product_key_memory import PKM
from reformer_lib.reversible import ReversibleSequence
from reformer_lib.scheduler import Scheduler

from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_

# constants
TOKEN_SELF_ATTN_VALUE = -5e4  # carefully set for half precision to work


# helper fns

def sort_key_val(t1, t2, dim=-1):
    values, indices = t1.sort(dim=dim)
    t2 = t2.expand_as(t1)
    return values, t2.gather(dim, indices), indices


def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))


def process_inputs_chunk(fn, chunks=1, dim=0):
    def inner_fn(*args, **kwargs):
        keys, values, len_args = kwargs.keys(), kwargs.values(), len(args)
        chunked_args = list(zip(*map(lambda x: x.chunk(chunks, dim=dim), list(args) + list(values))))
        all_args = map(lambda x: (x[:len_args], dict(zip(keys, x[len_args:]))), chunked_args)
        outputs = [fn(*c_args, **c_kwargs) for c_args, c_kwargs in all_args]

        # return tuple(map(lambda x: torch.cat(x, dim=dim), zip(*outputs)))

        def cat_fn(x):
            if x[0] is not None:
                return torch.cat(x, dim=dim)
            else:
                return None

        return tuple(map(cat_fn, zip(*outputs)))

    return inner_fn


def chunked_sum(tensor, chunks=1):
    *orig_size, last_dim = tensor.shape
    tensor = tensor.reshape(-1, last_dim)
    summed_tensors = [c.sum(dim=-1) for c in tensor.chunk(chunks, dim=0)]
    return torch.cat(summed_tensors, dim=0).reshape(orig_size)


def default(val, default_val):
    return default_val if val is None else val


def cast_tuple(x):
    return x if isinstance(x, tuple) else (x,)


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, **kwargs):
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


def cosine_similarity(x1, x2, dim=1, eps=1e-6):
    r"""Returns cosine similarity between x1 and x2, computed along dim.

    Args:
        x1 (Variable): First input.
        x2 (Variable): Second input (of size matching x1).
        dim (int, optional): Dimension of vectors. Default: 1
        eps (float, optional): Small value to avoid division by zero. Default: 1e-8

    Shape:
        - Input: :math:`(\ast_1, D, \ast_2)` where D is at position `dim`.
        - Output: :math:`(\ast_1, \ast_2)` where 1 is at position `dim`.
    """
    w1 = torch.norm(x1 + eps, 2, dim, keepdim=True)
    w2 = torch.norm(x2 + eps, 2, dim, keepdim=True)
    x1 /= w1.clamp(min=eps)
    x2 /= w2.clamp(min=eps)
    w12 = torch.sum(x1 * x2, dim)
    return w12.squeeze()


def cache_method_decorator(cache_attr, cache_namespace, reexecute=False):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, key_namespace=None, fetch=False, set_cache=True, **kwargs):
            namespace_str = str(default(key_namespace, ''))
            _cache = getattr(self, cache_attr)
            _keyname = f'{cache_namespace}:{namespace_str}'

            if fetch:
                val = _cache[_keyname]
                if reexecute:
                    fn(self, *args, **kwargs)
            else:
                val = fn(self, *args, **kwargs)
                if set_cache:
                    setattr(self, cache_attr, {**_cache, **{_keyname: val}})
            return val

        return wrapper

    return inner_fn


def expand_dim(dim, k, t):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)


def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]


# helper classes

class MatrixMultiply(nn.Module):
    def __init__(self, tensor, transpose=False, normalize=False):
        super().__init__()
        self.tensor = tensor
        self.transpose = transpose
        self.normalize = normalize

    def forward(self, x):
        tensor = self.tensor
        if self.normalize:
            tensor = F.normalize(tensor, dim=-1)
        if self.transpose:
            tensor = tensor.t()
        return x @ tensor


class ReZero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.g = nn.Parameter(torch.zeros(1))
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.g


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x / n * self.g


class PreNorm(nn.Module):
    def __init__(self, norm_class, dim, fn):
        super().__init__()
        self.norm = norm_class(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim=-1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        chunks = x.chunk(self.chunks, dim=self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim=self.dim)


# LSH attention as described in https://openreview.net/pdf?id=rkgNKkHtvB
# adapted from trax, stripped to what paper said needed to work
# namely that buckets need to be at least 64 with 8 rounds of hashing
# https://github.com/google/trax/blob/master/trax/layers/research/efficient_attention.py#L442

# +
class LSHAttention(nn.Module):
    def __init__(self,
                 dropout=0.,
                 bucket_size=64,
                 n_hashes=8,
                 causal=False,
                 allow_duplicate_attention=True,
                 attend_across_buckets=True,
                 rehash_each_round=True,
                 drop_for_hash_rate=0.0,
                 random_rotations_per_head=False,
                 return_attn=False,
                 store_stats=False):
        super().__init__()
        if dropout >= 1.0:
            raise ValueError('Dropout rates must be lower than 1.')

        self.dropout = nn.Dropout(dropout)
        self.dropout_for_hash = nn.Dropout(drop_for_hash_rate)

        #         self.rotations = nn.Linear(64, 128, bias=False)

        assert rehash_each_round or allow_duplicate_attention, (
            'The setting {allow_duplicate_attention=False, rehash_each_round=False}'
            ' is not implemented.')

        self.causal = causal
        self.bucket_size = bucket_size

        self.n_hashes = n_hashes

        self._allow_duplicate_attention = allow_duplicate_attention
        self._attend_across_buckets = attend_across_buckets
        self._rehash_each_round = rehash_each_round
        self._random_rotations_per_head = random_rotations_per_head

        # will expend extra computation to return attention matrix
        self._return_attn = return_attn

        # cache buckets for reversible network, reported by authors to make Reformer work at depth
        self._cache = {}

        self.store_stats = store_stats
        self.mean_dp = 0.0
        self.stat_count = 0

    # @cache_method_decorator('_cache', 'buckets', reexecute=True)
    def hash_vectors(self, n_buckets, vecs, rotations=None):
        batch_size = vecs.shape[0]
        device = vecs.device

        # See https://arxiv.org/pdf/1509.02897.pdf
        # We sample a different random rotation for each round of hashing to
        # decrease the probability of hash misses.
        assert n_buckets % 2 == 0

        rot_size = n_buckets

        rotations_shape = (
            batch_size if self._random_rotations_per_head else 1,
            vecs.shape[-1],
            self.n_hashes if self._rehash_each_round else 1,
            rot_size // 2)

        # add rotations
        # random_rotations = torch.randn(rotations_shape, dtype=vecs.dtype, device=device).expand(batch_size, -1, -1, -1)

        if rotations is None:
            random_rotations = torch.randn(rotations_shape, dtype=vecs.dtype, device=device).expand(batch_size, -1, -1,
                                                                                                    -1)
        else:
            if rotations.size(-1) == rotations_shape[-1]:
                random_rotations = rotations
            elif rotations.size(-1) < rotations_shape[-1]:
                complement_shape = (
                    batch_size if self._random_rotations_per_head else 1,
                    vecs.shape[-1],
                    self.n_hashes if self._rehash_each_round else 1,
                    rot_size // 2 - rotations.size(-1))

                tmp = torch.randn(complement_shape, dtype=vecs.dtype, device=device).expand(batch_size, -1, -1, -1)
                random_rotations = torch.cat([rotations, tmp], dim=-1)
            else:
                random_rotations = rotations[:, :, :, torch.randperm(rotations.size(-1))[:rotations_shape[-1]]]

        dropped_vecs = self.dropout_for_hash(vecs)
        rotated_vecs = torch.einsum('btf,bfhi->bhti', dropped_vecs, random_rotations)

        if self._rehash_each_round:
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            buckets = torch.argmax(rotated_vecs, dim=-1)
            # buckets is now (self.n_hashes, seqlen). Next we add offsets so that
            # bucket numbers from different hashing rounds don't overlap.
            offsets = torch.arange(self.n_hashes, device=device)
            offsets = torch.reshape(offsets * n_buckets, (1, -1, 1))
            buckets = torch.reshape(buckets + offsets, (batch_size, -1,))
        else:
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            # In this configuration, we map each item to the top self.n_hashes buckets
            rotated_vecs = torch.squeeze(rotated_vecs, 0)
            bucket_range = torch.arange(rotated_vecs.shape[-1], device=device)
            bucket_range = torch.reshape(bucket_range, (1, -1))
            bucket_range = bucket_range.expand_as(rotated_vecs.shape)

            _, buckets = sort_key_val(rotated_vecs, bucket_range, dim=-1)
            buckets = buckets[:, -self.n_hashes:]

            h, *_ = buckets.shape
            buckets = torch.reshape(buckets.permute((*_, h)), (-1,))

        return buckets

    def forward(self, qk, v, query_len=None, input_mask=None, input_attn_mask=None, rotations=None,
                triplet_examples=False, **kwargs):
        batch_size, seqlen, dim, device = *qk.shape, qk.device

        query_len = default(query_len, seqlen)
        is_reverse = kwargs.pop('_reverse', False)
        depth = kwargs.pop('_depth', None)

        assert seqlen % (
                self.bucket_size * 2) == 0, f'Sequence length ({seqlen}) needs to be divisible by target bucket size  x 2 - {self.bucket_size * 2}'

        n_buckets = seqlen // self.bucket_size
        # buckets = self.hash_vectors(n_buckets, qk, key_namespace=depth, fetch=is_reverse, set_cache=self.training)

        # add customized rotations
        # buckets = self.hash_vectors(n_buckets, qk, key_namespace=depth, fetch=is_reverse, set_cache=self.training, rotations=rotations)
        buckets = self.hash_vectors(n_buckets, qk, rotations=rotations)

        max_idxs = buckets.reshape(batch_size, -1, seqlen)

        # We use the same vector as both a query and a key.
        assert int(buckets.shape[1]) == self.n_hashes * seqlen

        total_hashes = self.n_hashes

        ticker = torch.arange(total_hashes * seqlen, device=device).unsqueeze(0).expand_as(buckets)
        buckets_and_t = seqlen * buckets + (ticker % seqlen)
        buckets_and_t = buckets_and_t.detach()

        # Hash-based sort ("s" at the start of variable names means "sorted")
        sbuckets_and_t, sticker, indices = sort_key_val(buckets_and_t, ticker, dim=-1)
        _, undo_sort = sticker.sort(dim=-1)
        del ticker

        sbuckets_and_t = sbuckets_and_t.detach()
        sticker = sticker.detach()
        undo_sort = undo_sort.detach()

        st = (sticker % seqlen)
        sqk = batched_index_select(qk, st)
        sv = batched_index_select(v, st)

        # Split off a "bin" axis so that attention only occurs within chunks.
        chunk_size = total_hashes * n_buckets
        bq_t = bkv_t = torch.reshape(st, (batch_size, chunk_size, -1))
        bqk = torch.reshape(sqk, (batch_size, chunk_size, -1, dim))
        bv = torch.reshape(sv, (batch_size, chunk_size, -1, dim))

        # Hashing operates on unit-length vectors. Unnormalized query vectors are
        # fine because they effectively provide a learnable temperature for the
        # attention softmax, but normalizing keys is needed so that similarity for
        # the purposes of attention correctly corresponds to hash locality.
        bq = bqk
        bk = F.normalize(bqk, p=2, dim=-1).type_as(bq)

        # Allow each chunk to attend within itself, and also one chunk back. Chunk
        # boundaries might occur in the middle of a sequence of items from the
        # same bucket, so this increases the chances of attending to relevant items.
        def look_one_back(x):
            x_extra = torch.cat([x[:, -1:, ...], x[:, :-1, ...]], dim=1)
            return torch.cat([x, x_extra], dim=2)

        bk = look_one_back(bk)
        bv = look_one_back(bv)
        bkv_t = look_one_back(bkv_t)

        # Dot-product attention.
        dots = torch.einsum('bhie,bhje->bhij', bq, bk) * (dim ** -0.5)
        if triplet_examples:
            min_samples = dots.argmin(dim=-1).detach()
        masked_value = max_neg_value(dots)

        # Mask for post qk attention logits of the input sequence
        if input_attn_mask is not None:
            input_attn_mask = F.pad(input_attn_mask,
                                    (0, seqlen - input_attn_mask.shape[-1], 0, seqlen - input_attn_mask.shape[-2]),
                                    value=True)
            dot_attn_indices = ((bq_t * seqlen)[:, :, :, None] + bkv_t[:, :, None, :])
            input_attn_mask = input_attn_mask.reshape(batch_size, -1)
            dot_attn_indices = dot_attn_indices.reshape(batch_size, -1)
            mask = input_attn_mask.gather(1, dot_attn_indices).reshape_as(dots)
            dots.masked_fill_(~mask, masked_value)
            del mask

        # Input mask for padding in variable lengthed sequences
        if input_mask is not None:
            input_mask = F.pad(input_mask, (0, seqlen - input_mask.shape[1]), value=True)
            mq = input_mask.gather(1, st).reshape((batch_size, chunk_size, -1))
            mkv = look_one_back(mq)
            mask = mq[:, :, :, None] * mkv[:, :, None, :]
            dots.masked_fill_(~mask, masked_value)
            del mask

        # Causal masking
        if self.causal:
            mask = bq_t[:, :, :, None] < bkv_t[:, :, None, :]
            if seqlen > query_len:
                mask = mask & (bkv_t[:, :, None, :] < query_len)
            dots.masked_fill_(mask, masked_value)
            del mask

        # Mask out attention to self except when no other targets are available.
        self_mask = bq_t[:, :, :, None] == bkv_t[:, :, None, :]
        dots.masked_fill_(self_mask, TOKEN_SELF_ATTN_VALUE)
        del self_mask

        # Mask out attention to other hash buckets.
        if not self._attend_across_buckets:
            bq_buckets = bkv_buckets = torch.reshape(sbuckets_and_t // seqlen, (batch_size, chunk_size, -1))
            bkv_buckets = look_one_back(bkv_buckets)
            bucket_mask = bq_buckets[:, :, :, None] != bkv_buckets[:, :, None, :]
            dots.masked_fill_(bucket_mask, masked_value)
            del bucket_mask

        # Don't double-count query-key pairs across multiple rounds of hashing.
        # There are two possible strategies here. (1) The default is to count how
        # many times a query-key pair is repeated, and to lower its log-prob
        # correspondingly at each repetition. (2) When hard_k is set, the code
        # instead masks all but the first occurence of each query-key pair.
        if not self._allow_duplicate_attention:
            locs1 = undo_sort // bq_t.shape[-1]
            locs2 = (locs1 + 1) % chunk_size
            if not self._attend_across_buckets:
                locs1 = buckets * chunk_size + locs1
                locs2 = buckets * chunk_size + locs2
            locs = torch.cat([
                torch.reshape(locs1, (batch_size, total_hashes, seqlen)),
                torch.reshape(locs2, (batch_size, total_hashes, seqlen)),
            ], 1).permute((0, 2, 1))

            slocs = batched_index_select(locs, st)
            b_locs = torch.reshape(slocs, (batch_size, chunk_size, -1, 2 * total_hashes))

            b_locs1 = b_locs[:, :, :, None, :total_hashes]

            bq_locs = b_locs1.expand(b_locs.shape[:3] + (2, total_hashes))
            bq_locs = torch.reshape(bq_locs, b_locs.shape)
            bkv_locs = look_one_back(b_locs)

            dup_counts = (bq_locs[:, :, :, None, :] == bkv_locs[:, :, None, :, :])
            # for memory considerations, chunk summation of last dimension for counting duplicates
            dup_counts = chunked_sum(dup_counts, chunks=(total_hashes * batch_size))
            dup_counts = dup_counts.detach()
            assert dup_counts.shape == dots.shape
            dots = dots - torch.log(dup_counts + 1e-9)
            del dup_counts

        with torch.no_grad():
            if self.store_stats:
                computed_mean = dots.detach()
                mean = torch.mean(computed_mean[computed_mean > TOKEN_SELF_ATTN_VALUE + 1])
                self.mean_dp = mean.item()

                self.stat_count += 1

        # Softmax.
        dots_logsumexp = torch.logsumexp(dots, dim=-1, keepdim=True)
        dots = torch.exp(dots - dots_logsumexp).type_as(dots)

        dropped_dots = self.dropout(dots)

        bo = torch.einsum('buij,buje->buie', dropped_dots, bv)
        so = torch.reshape(bo, (batch_size, -1, dim))
        slogits = torch.reshape(dots_logsumexp, (batch_size, -1,))

        # compute pos/neg examples
        if triplet_examples:
            with torch.no_grad():
                max_samples = dots.argmax(dim=-1)

                max_ind = torch.gather(bkv_t, -1, max_samples).view_as(st)
                min_ind = torch.gather(bkv_t, -1, min_samples).view_as(st)

                pos_vectors = batched_index_select(qk, max_ind).detach()
                neg_vectors = batched_index_select(qk, min_ind).detach()
        else:
            pos_vectors = None
            neg_vectors = None

        # unsort logits
        o = batched_index_select(so, undo_sort)
        logits = slogits.gather(1, undo_sort)

        o = torch.reshape(o, (batch_size, total_hashes, seqlen, dim))
        logits = torch.reshape(logits, (batch_size, total_hashes, seqlen, 1))

        if query_len != seqlen:
            query_slice = (slice(None), slice(None), slice(0, query_len))
            o, logits = o[query_slice], logits[query_slice]

        probs = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))
        out = torch.sum(o * probs, dim=1)

        attn = torch.empty(0, device=device)

        # return unsorted attention weights
        if self._return_attn:
            attn_unsort = ((bq_t * seqlen)[:, :, :, None] + bkv_t[:, :, None, :])
            attn_unsort = attn_unsort.view(batch_size * total_hashes, -1).long()
            unsorted_dots = torch.zeros(batch_size * total_hashes, seqlen * seqlen, device=device)
            unsorted_dots.scatter_add_(1, attn_unsort, dots.view_as(attn_unsort))
            del attn_unsort
            unsorted_dots = unsorted_dots.reshape(batch_size, total_hashes, seqlen, seqlen)
            attn = torch.sum(unsorted_dots[:, :, 0:query_len, :] * probs, dim=1)

        # return output, attention matrix, and bucket distribution
        return out, attn, buckets, sqk.detach(), pos_vectors, neg_vectors


# customized training for hash functions

class TripletLSHAttention(LSHAttention):
    def __init__(self,
                 alpha=1.0,  # triplet loss margin
                 dim=512,  # embedding dimension
                 seq_len=1024,
                 heads=8,  # attention heads
                 dropout=0.,
                 bucket_size=64,
                 n_hashes=8,
                 causal=False,
                 allow_duplicate_attention=True,
                 attend_across_buckets=True,
                 rehash_each_round=True,
                 drop_for_hash_rate=0.0,
                 random_rotations_per_head=False,
                 return_attn=False,
                 triplet_chunks=None,
                 store_stats=False
                 ):
        super().__init__(dropout=dropout,
                         bucket_size=bucket_size,
                         n_hashes=n_hashes,
                         causal=causal,
                         allow_duplicate_attention=allow_duplicate_attention,
                         attend_across_buckets=attend_across_buckets,
                         rehash_each_round=rehash_each_round,
                         drop_for_hash_rate=drop_for_hash_rate,
                         random_rotations_per_head=random_rotations_per_head,
                         return_attn=return_attn,
                         store_stats=store_stats)
        self.alpha = alpha
        self.seq_len = seq_len
        self.heads = heads
        n_buckets = self.seq_len // bucket_size
        # buckets_dim = n_buckets // 2
        buckets_dim = n_buckets
        if self._rehash_each_round:
            buckets_dim *= n_hashes
        self.rotations = nn.Linear(dim // self.heads, buckets_dim, bias=False)

        # number of chunks to split up computation of pos/neg examples for triplet loss
        self.triplet_chunks = default(triplet_chunks, dim)

    def reset_rotations(self):
        self.rotations.reset_parameters()

    def extract_rotations(self, batch_size):
        n_buckets = self.seq_len // self.bucket_size
        rotations = self.rotations.weight.t().detach()  # dim x (buckets * n_hashes / 2)
        # rotations = rotations[:, torch.randperm(rotations.size(-1))[:rotations.size(-1) // 2]]
        rotations = torch.reshape(rotations, (-1, self.n_hashes, n_buckets))
        rotations = rotations.unsqueeze(0).expand(batch_size, -1, -1, -1)
        return rotations

    def triplet_forward(self,
                        x,  # input
                        p,  # positive example
                        n,  # negative example
                        ):
        '''
        Given inputs, positive and negative examples, compute the
        triplet loss given by cosine similarity
        '''
        x = x.detach()
        p = p.detach()
        n = n.detach()
        emb_x = self.rotations(x)
        emb_p = self.rotations(p)
        emb_n = self.rotations(n)

        # cosine similarity
        sim_xp = F.cosine_similarity(emb_x, emb_p, dim=-1, eps=1e-6)
        sim_xn = F.cosine_similarity(emb_x, emb_n, dim=-1, eps=1e-6)

        # distance in radians
        # dis_xp = 1 - torch.acos(sim_xp)/pi
        # dis_xn = 1 - torch.acos(sim_xn)/pi
        dis_xp = 1 - sim_xp
        dis_xn = 1 - sim_xn
        triplet_loss = dis_xp - dis_xn + self.alpha

        triplet_loss = torch.mean(torch.max(triplet_loss,
                                            torch.zeros(triplet_loss.size()).to(x.device)))
        if torch.isnan(triplet_loss):
            print("nan!")

        return triplet_loss

    def forward(self, qk, v, query_len=None, input_mask=None, printgrad=False,
                triplet_examples=False, **kwargs):
        batch_size, seqlen, dim = qk.shape
        n_buckets = self.seq_len // self.bucket_size
        rotations = self.extract_rotations(batch_size)
        # self.rotations.reset_parameters()
        out, attn, buckets, emb_x, pos, neg = super().forward(qk, v,
                                                              query_len=query_len,
                                                              input_mask=input_mask,
                                                              rotations=rotations,
                                                              printgrad=printgrad,
                                                              triplet_examples=triplet_examples)
        return out, attn, buckets, emb_x, pos, neg


# simple full attention
class FullQKAttention(nn.Module):
    def __init__(self, causal=False, dropout=0.):
        super().__init__()
        self.causal = causal
        self.dropout = nn.Dropout(dropout)
        self.attn = None

    def forward(self, qk, v, query_len=None, input_mask=None, input_attn_mask=None, **kwargs):
        b, seq_len, dim = qk.shape
        query_len = default(query_len, seq_len)
        t = query_len

        q = qk[:, 0:query_len]
        qk = F.normalize(qk, 2, dim=-1).type_as(q)

        dot = torch.einsum('bie,bje->bij', q, qk) * (dim ** -0.5)

        # qk attention requires tokens not attend to self
        i = torch.arange(t)
        dot[:, i, i] = TOKEN_SELF_ATTN_VALUE
        masked_value = max_neg_value(dot)

        # Input mask for padding in variable lengthed sequences
        if input_mask is not None:
            mask = input_mask[:, 0:query_len, None] * input_mask[:, None, :]
            mask = F.pad(mask, (0, seq_len - mask.shape[-1]), value=True)
            dot.masked_fill_(~mask, masked_value)

        # Mask for post qk attention logits of the input sequence
        if input_attn_mask is not None:
            input_attn_mask = F.pad(input_attn_mask, (0, seq_len - input_attn_mask.shape[-1]), value=True)
            dot.masked_fill_(~input_attn_mask, masked_value)

        if self.causal:
            i, j = torch.triu_indices(t, t, 1)
            dot[:, i, j] = masked_value

        dot = dot.softmax(dim=-1)
        self.attn = dot.detach()
        dot = self.dropout(dot)

        out = torch.einsum('bij,bje->bie', dot, v)

        return out, dot, torch.empty(0), None, None, None


class LSHSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, bucket_size=64, n_hashes=8, causal=False, dim_head=None, attn_chunks=1,
                 random_rotations_per_head=False, attend_across_buckets=True, allow_duplicate_attention=True,
                 num_mem_kv=0, one_value_head=False, use_full_attn=False, full_attn_thres=None, return_attn=False,
                 post_attn_dropout=0., dropout=0., n_local_attn_heads=0, attn_type='lsh', max_seq_len=None,
                 alpha=1.0, triplet_chunks=None, **kwargs):
        super().__init__()
        assert dim_head or (dim % heads) == 0, 'dimensions must be divisible by number of heads'
        assert n_local_attn_heads < heads, 'local attention heads must be less than number of heads'

        dim_head = default(dim_head, dim // heads)
        dim_heads = dim_head * heads

        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.attn_chunks = default(attn_chunks, 1)

        self.v_head_repeats = (heads if one_value_head else 1)
        v_dim = dim_heads // self.v_head_repeats

        self.toqk = nn.Linear(dim, dim_heads, bias=False)
        self.tov = nn.Linear(dim, v_dim, bias=False)
        self.to_out = nn.Linear(dim_heads, dim)

        self.bucket_size = bucket_size

        self.attn_type = attn_type
        if self.attn_type == 'triplet':
            self.lsh_attn = TripletLSHAttention(alpha=alpha, dim=self.dim, seq_len=max_seq_len, heads=self.heads,
                                                bucket_size=bucket_size, n_hashes=n_hashes, causal=causal,
                                                random_rotations_per_head=random_rotations_per_head,
                                                attend_across_buckets=attend_across_buckets,
                                                allow_duplicate_attention=allow_duplicate_attention,
                                                return_attn=return_attn, triplet_chunks=triplet_chunks, **kwargs)
            # init scheduler
            K = 1
            L = 10
            thresh = 0.1
            self.scheduler = Scheduler(self.toqk.weight, dim, K, L, thresh)

        else:
            self.lsh_attn = LSHAttention(bucket_size=bucket_size, n_hashes=n_hashes, causal=causal,
                                         random_rotations_per_head=random_rotations_per_head,
                                         attend_across_buckets=attend_across_buckets,
                                         allow_duplicate_attention=allow_duplicate_attention, return_attn=return_attn,
                                         dropout=dropout, **kwargs)

        self.full_attn = FullQKAttention(causal=causal, dropout=dropout)
        self.post_attn_dropout = nn.Dropout(post_attn_dropout)

        self.use_full_attn = use_full_attn
        self.full_attn_thres = default(full_attn_thres, bucket_size)

        self.num_mem_kv = num_mem_kv
        self.mem_kv = nn.Parameter(torch.randn(1, num_mem_kv, dim, requires_grad=True)) if num_mem_kv > 0 else None

        self.n_local_attn_heads = n_local_attn_heads
        self.local_attn = LocalAttention(window_size=bucket_size * 2, causal=causal, dropout=dropout, shared_qk=True,
                                         look_forward=(1 if not causal else 0))

        self.callback = None
        self.attn = None
        self.triplet_loss = 0.0
        self._reset_parameters()

    def _reset_parameters(self):
        # if self._qkv_same_embed_dim:
        xavier_uniform_(self.toqk.weight)
        xavier_uniform_(self.tov.weight)
        xavier_uniform_(self.to_out.weight)
        # else:
        #     xavier_uniform_(self.q_proj_weight)
        #     xavier_uniform_(self.k_proj_weight)
        #     xavier_uniform_(self.v_proj_weight)

        # if self.in_proj_bias is not None:
        # constant_(self.toqk.bias, 0.)
        # constant_(self.tov.bias, 0.)
        # constant_(self.to_out.bias, 0.)
        # if self.bias_k is not None:
        #     xavier_normal_(self.bias_k)
        # if self.bias_v is not None:
        #     xavier_normal_(self.bias_v)

    def forward(self, x, keys=None, input_mask=None, input_attn_mask=None, context_mask=None, calc_triplet=False,
                **kwargs):
        device, dtype = x.device, x.dtype
        b, t, e, h, dh, m, l_h = *x.shape, self.heads, self.dim_head, self.num_mem_kv, self.n_local_attn_heads

        mem_kv = default(self.mem_kv, torch.empty(b, 0, e, dtype=dtype, device=device))
        mem = mem_kv.expand(-1, m, -1)

        keys = default(keys, torch.empty(b, 0, e, dtype=dtype, device=device))
        c = keys.shape[1]

        kv_len = t + m + c
        use_full_attn = self.use_full_attn or kv_len <= self.full_attn_thres

        x = torch.cat((x, mem, keys), dim=1)
        qk = self.toqk(x)
        v = self.tov(x)
        v = v.repeat(1, 1, self.v_head_repeats)

        def merge_heads(v):
            return v.view(b, kv_len, h, -1).transpose(1, 2)

        def split_heads(v):
            return v.view(b, h, t, -1).transpose(1, 2).contiguous()

        merge_batch_and_heads = partial(merge_dims, 0, 1)

        qk, v = map(merge_heads, (qk, v))

        has_local = l_h > 0
        lsh_h = h - l_h

        split_index_fn = partial(split_at_index, 1, l_h)
        (lqk, qk), (lv, v) = map(split_index_fn, (qk, v))
        lqk, qk, lv, v = map(merge_batch_and_heads, (lqk, qk, lv, v))

        masks = {}
        if input_mask is not None or context_mask is not None:
            default_mask = torch.tensor([True], device=device)
            i_mask = default(input_mask, default_mask.expand(b, t))
            m_mask = default_mask.expand(b, m)
            c_mask = default(context_mask, default_mask.expand(b, c))
            mask = torch.cat((i_mask, m_mask, c_mask), dim=1)
            mask = merge_batch_and_heads(expand_dim(1, lsh_h, mask))
            masks['input_mask'] = mask

        if input_attn_mask is not None:
            input_attn_mask = merge_batch_and_heads(expand_dim(1, lsh_h, input_attn_mask))
            masks['input_attn_mask'] = input_attn_mask

        attn_fn = self.lsh_attn if not use_full_attn else self.full_attn

        # update rotations
        # if calc_triplet:
        #     if not self.scheduler.detect_change(self.toqk.weight):
        #         calc_triplet = False
        return_triplet_examples = (self.attn_type in ['triplet', 'simhash']) and calc_triplet and not use_full_attn
        partial_attn_fn = partial(attn_fn, query_len=t, input_mask=input_mask,
                                  triplet_examples=return_triplet_examples)

        attn_fn_in_chunks = process_inputs_chunk(partial_attn_fn, chunks=self.attn_chunks)
        out, attn, buckets, emb_x, pos, neg = attn_fn_in_chunks(qk, v, **masks)

        if self.callback is not None:
            self.callback(attn.reshape(b, lsh_h, t, -1), buckets.reshape(b, lsh_h, -1))

        if return_triplet_examples:
            def chunked_loss(fn, *args, chunks=1, dim=0):
                chunked_inputs = list(map(lambda x: x.chunk(chunks, dim=dim), args))
                outputs = [fn(*inputs) for inputs in zip(*chunked_inputs)]
                return sum(outputs)

            triplet_loss = chunked_loss(self.lsh_attn.triplet_forward,
                                        emb_x, pos, neg,
                                        chunks=self.attn_chunks, dim=1)

            if self.triplet_loss is None:
                self.triplet_loss = triplet_loss
            else:
                self.triplet_loss += triplet_loss

        if has_local:
            lqk, lv = lqk[:, :t], lv[:, :t]
            local_out = self.local_attn(lqk, lqk, lv, input_mask=input_mask)
            local_out = local_out.reshape(b, l_h, t, -1)
            out = out.reshape(b, lsh_h, t, -1)
            out = torch.cat((local_out, out), dim=1)

        out = split_heads(out).view(b, t, -1)

        self.attn = out.detach()

        out = self.to_out(out)
        return self.post_attn_dropout(out)


# feed forward
class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0., activation=None, glu=False):
        super().__init__()
        activation = default(activation, GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x


# positional embeddings
class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)


class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb[None, :, :]


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# reformer lm
class Reformer_tune(nn.Module):
    def __init__(self, dim, depth, max_seq_len, heads=8, dim_head=None, bucket_size_list=[], n_hashes_list=[],
                 ff_chunks=100,
                 attn_chunks=None, causal=False, weight_tie=False, lsh_dropout=0., ff_dropout=0., ff_activation=None,
                 ff_mult=4, ff_glu=False, post_attn_dropout=0., layer_dropout=0., lsh_attend_across_buckets=True,
                 lsh_allow_duplicate_attention=True, random_rotations_per_head=False, twin_attention=False,
                 use_scale_norm=False, use_rezero=False, use_full_attn=False, full_attn_thres=0, reverse_thres=0,
                 num_mem_kv=0, one_value_head=False, n_local_attn_heads=0, pkm_layers=tuple(), pkm_num_keys=128,
                 attn_type_list=[], store_stats=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        assert len(bucket_size_list) == depth
        assert len(n_hashes_list) == depth
        assert len(attn_type_list) == depth

        self.bucket_size_list = bucket_size_list
        self.num_mem_kv = num_mem_kv

        self.twin_attention = twin_attention
        self.full_attn_thres = full_attn_thres

        get_ff = lambda: Chunk(ff_chunks,
                               FeedForward(dim, dropout=ff_dropout, activation=ff_activation, mult=ff_mult, glu=ff_glu),
                               along_dim=-2)
        get_pkm = lambda: PKM(dim, num_keys=pkm_num_keys)

        if weight_tie:
            get_attn = lambda: LSHSelfAttention(dim, heads, bucket_size_list[0], n_hashes_list[0], causal=causal,
                                                dim_head=dim_head,
                                                dropout=lsh_dropout, post_attn_dropout=post_attn_dropout,
                                                attn_chunks=attn_chunks,
                                                allow_duplicate_attention=lsh_allow_duplicate_attention,
                                                attend_across_buckets=lsh_attend_across_buckets,
                                                random_rotations_per_head=random_rotations_per_head,
                                                num_mem_kv=num_mem_kv,
                                                use_full_attn=use_full_attn, full_attn_thres=full_attn_thres,
                                                one_value_head=one_value_head, n_local_attn_heads=n_local_attn_heads,
                                                max_seq_len=max_seq_len, attn_type=attn_type_list[0],
                                                store_stats=store_stats)
            get_attn, get_ff, get_pkm = map(cache_fn, (get_attn, get_ff, get_pkm))

        blocks = []

        norm_type = ScaleNorm if use_scale_norm else nn.LayerNorm

        residual_fn_wrapper = ReZero if use_rezero else partial(PreNorm, norm_type, dim)

        for ind in range(depth):
            layer_num = ind + 1
            use_pkm = layer_num in cast_tuple(pkm_layers)
            parallel_net = None

            attn = LSHSelfAttention(dim, heads, bucket_size_list[ind], n_hashes_list[ind], causal=causal,
                                    dim_head=dim_head,
                                    dropout=lsh_dropout, post_attn_dropout=post_attn_dropout,
                                    attn_chunks=attn_chunks,
                                    allow_duplicate_attention=lsh_allow_duplicate_attention,
                                    attend_across_buckets=lsh_attend_across_buckets,
                                    random_rotations_per_head=random_rotations_per_head, num_mem_kv=num_mem_kv,
                                    use_full_attn=use_full_attn, full_attn_thres=full_attn_thres,
                                    one_value_head=one_value_head, n_local_attn_heads=n_local_attn_heads,
                                    max_seq_len=max_seq_len, attn_type=attn_type_list[ind], store_stats=store_stats)

            if use_pkm:
                parallel_net = get_pkm()
            elif twin_attention:
                parallel_net = get_attn()
            else:
                parallel_net = get_ff()

            f = residual_fn_wrapper(attn)
            g = residual_fn_wrapper(parallel_net)

            blocks.append(nn.ModuleList([f, g]))

        self.layers = ReversibleSequence(nn.ModuleList(blocks), layer_dropout=layer_dropout,
                                         reverse_thres=reverse_thres, send_signal=True)
        self.layer_modules = list(chain(*[[m[0], m[1]] for m in blocks]))

    def forward(self, x, **kwargs):
        x = torch.cat([x, x], dim=-1)
        arg_route = (True, self.twin_attention)
        x = self.layers(x, arg_route=arg_route, **kwargs)
        return torch.stack(x.chunk(2, dim=-1)).mean(dim=0)


# reformer lm
class Reformer(nn.Module):
    def __init__(self, dim, depth, max_seq_len, heads=8, dim_head=None, bucket_size=64, n_hashes=8, ff_chunks=100,
                 attn_chunks=None, causal=False, weight_tie=False, lsh_dropout=0., ff_dropout=0., ff_activation=None,
                 ff_mult=4, ff_glu=False, post_attn_dropout=0., layer_dropout=0., lsh_attend_across_buckets=True,
                 lsh_allow_duplicate_attention=True, random_rotations_per_head=False, twin_attention=False,
                 use_scale_norm=False, use_rezero=False, use_full_attn=False, full_attn_thres=0, reverse_thres=0,
                 num_mem_kv=0, one_value_head=False, n_local_attn_heads=0, pkm_layers=tuple(), pkm_num_keys=128,
                 attn_type='lsh', store_stats=False):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.bucket_size = bucket_size
        self.num_mem_kv = num_mem_kv

        self.twin_attention = twin_attention
        self.full_attn_thres = full_attn_thres

        get_attn = lambda: LSHSelfAttention(dim, heads, bucket_size, n_hashes, causal=causal, dim_head=dim_head,
                                            dropout=lsh_dropout, post_attn_dropout=post_attn_dropout,
                                            attn_chunks=attn_chunks,
                                            allow_duplicate_attention=lsh_allow_duplicate_attention,
                                            attend_across_buckets=lsh_attend_across_buckets,
                                            random_rotations_per_head=random_rotations_per_head, num_mem_kv=num_mem_kv,
                                            use_full_attn=use_full_attn, full_attn_thres=full_attn_thres,
                                            one_value_head=one_value_head, n_local_attn_heads=n_local_attn_heads,
                                            max_seq_len=max_seq_len, attn_type=attn_type, store_stats=store_stats)
        get_ff = lambda: Chunk(ff_chunks,
                               FeedForward(dim, dropout=ff_dropout, activation=ff_activation, mult=ff_mult, glu=ff_glu),
                               along_dim=-2)
        get_pkm = lambda: PKM(dim, num_keys=pkm_num_keys)

        if weight_tie:
            get_attn, get_ff, get_pkm = map(cache_fn, (get_attn, get_ff, get_pkm))

        blocks = []

        norm_type = ScaleNorm if use_scale_norm else nn.LayerNorm

        residual_fn_wrapper = ReZero if use_rezero else partial(PreNorm, norm_type, dim)

        for ind in range(depth):
            layer_num = ind + 1
            use_pkm = layer_num in cast_tuple(pkm_layers)
            parallel_net = None

            attn = get_attn()

            if use_pkm:
                parallel_net = get_pkm()
            elif twin_attention:
                parallel_net = get_attn()
            else:
                parallel_net = get_ff()

            f = residual_fn_wrapper(attn)
            g = residual_fn_wrapper(parallel_net)

            blocks.append(nn.ModuleList([f, g]))

        self.layers = ReversibleSequence(nn.ModuleList(blocks), layer_dropout=layer_dropout,
                                         reverse_thres=reverse_thres, send_signal=True)
        self.layer_modules = list(chain(*[[m[0], m[1]] for m in blocks]))

    def forward(self, x, **kwargs):
        x = torch.cat([x, x], dim=-1)
        arg_route = (True, self.twin_attention)
        x = self.layers(x, arg_route=arg_route, **kwargs)
        return torch.stack(x.chunk(2, dim=-1)).mean(dim=0)


class ReformerLM_tune(nn.Module):
    def __init__(self, num_tokens, dim, depth, max_seq_len, heads=8, dim_head=None, bucket_size_list=[],
                 n_hashes_list=[],
                 ff_chunks=100, attn_chunks=1, causal=False, weight_tie=False, lsh_dropout=0., ff_dropout=0., ff_mult=4,
                 ff_activation=None, ff_glu=False, post_attn_dropout=0., layer_dropout=0.,
                 random_rotations_per_head=False, twin_attention=False, use_scale_norm=False, use_rezero=False,
                 use_full_attn=False, full_attn_thres=0, reverse_thres=0, num_mem_kv=0, one_value_head=False,
                 emb_dim=None, return_embeddings=False, weight_tie_embedding=False, fixed_position_emb=False,
                 absolute_position_emb=False, axial_position_shape=None, n_local_attn_heads=0, pkm_layers=tuple(),
                 pkm_num_keys=128, attn_type_list=[], store_stats=False):
        super().__init__()
        emb_dim = default(emb_dim, dim)
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(num_tokens, emb_dim)

        self.to_model_dim = Identity() if emb_dim == dim else nn.Linear(emb_dim, dim)

        if absolute_position_emb:
            # self.pos_emb = PositionalEncoding(emb_dim, layer_dropout, max_seq_len)
            self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len)
        elif fixed_position_emb:
            self.pos_emb = FixedPositionalEmbedding(emb_dim)
        else:
            axial_position_shape = default(axial_position_shape,
                                           (max_seq_len // bucket_size_list[0], bucket_size_list[0]))
            self.pos_emb = AxialPositionalEmbedding(emb_dim, axial_position_shape)

        self.reformer = Reformer_tune(dim, depth, max_seq_len, heads=heads, dim_head=dim_head,
                                      bucket_size_list=bucket_size_list,
                                      n_hashes_list=n_hashes_list, ff_chunks=ff_chunks, attn_chunks=attn_chunks,
                                      causal=causal,
                                      weight_tie=weight_tie, lsh_dropout=lsh_dropout, ff_mult=ff_mult,
                                      ff_activation=ff_activation, ff_glu=ff_glu, ff_dropout=ff_dropout,
                                      post_attn_dropout=0., layer_dropout=layer_dropout,
                                      random_rotations_per_head=random_rotations_per_head,
                                      twin_attention=twin_attention,
                                      use_scale_norm=use_scale_norm, use_rezero=use_rezero, use_full_attn=use_full_attn,
                                      full_attn_thres=full_attn_thres, reverse_thres=reverse_thres,
                                      num_mem_kv=num_mem_kv,
                                      one_value_head=one_value_head, n_local_attn_heads=n_local_attn_heads,
                                      pkm_layers=pkm_layers, pkm_num_keys=pkm_num_keys, attn_type_list=attn_type_list,
                                      store_stats=store_stats)

        if return_embeddings:
            self.out = Identity()
            return

        self.out = nn.Sequential(
            nn.Linear(dim, emb_dim) if emb_dim != dim else Identity(),
            nn.Linear(emb_dim, num_tokens) if not weight_tie_embedding else MatrixMultiply(self.token_emb.weight,
                                                                                           transpose=True,
                                                                                           normalize=True)
        )
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.token_emb.weight.data.uniform_(-initrange, initrange)
        self.out[1].bias.data.zero_()
        self.out[1].weight.data.uniform_(-initrange, initrange)

    def forward(self, x, **kwargs):
        x = self.token_emb(x)
        x = x + self.pos_emb(x).type_as(x)
        # x = self.pos_emb(x)

        x = self.to_model_dim(x)
        x = self.reformer(x, **kwargs)
        return self.out(x)

    def clear_non_rotation_gradients(self):
        # clear gradients from triplet loss
        for i in range(len(self.reformer.layer_modules) // 2):
            f = self.reformer.layer_modules[2 * i].fn
            g = self.reformer.layer_modules[(2 * i) + 1].fn
            # only zero out toqk, tov, and to_out
            # leave gradients of rotations
            f.toqk.zero_grad()
            f.tov.zero_grad()
            f.to_out.zero_grad()
            g.zero_grad()

    def get_triplet_loss(self):
        total = 0
        for i in range(len(self.reformer.layer_modules) // 2):
            f = self.reformer.layer_modules[2 * i].fn
            if f.triplet_loss is not None:
                total += f.triplet_loss
        return total

    def update_simhash(self):
        for i in range(len(self.reformer.layer_modules) // 2):
            f = self.reformer.layer_modules[2 * i].fn
            attn_fn = f.lsh_attn
            if hasattr(attn_fn, 'simhash'):
                attn_fn.update_simhash()

    def reset_triplet(self):
        for i in range(len(self.reformer.layer_modules) // 2):
            f = self.reformer.layer_modules[2 * i].fn
            attn_fn = f.lsh_attn
            if hasattr(attn_fn, 'rotations'):
                attn_fn.reset_rotations()

    def get_statistics(self, batch_size):
        means = []
        for i in range(len(self.reformer.layer_modules) // 2):
            f = self.reformer.layer_modules[2 * i].fn
            attn_fn = f.lsh_attn
            means.append(attn_fn.mean_dp / attn_fn.stat_count)
            attn_fn.mean_dp = 0.0
            attn_fn.stat_count = 0
        return means

    def set_alpha(self, alpha):
        for i in range(len(self.reformer.layer_modules) // 2):
            f = self.reformer.layer_modules[2 * i].fn
            attn_fn = f.lsh_attn
            if hasattr(attn_fn, 'alpha'):
                attn_fn.alpha = alpha

    def clear_triplet_loss(self):
        for i in range(len(self.reformer.layer_modules) // 2):
            f = self.reformer.layer_modules[2 * i].fn
            if f.triplet_loss is not None:
                f.triplet_loss = None

    def save_triplet_params(self, prefix):
        for i in range(len(self.reformer.layer_modules) // 2):
            f = self.reformer.layer_modules[2 * i].fn
            weight = f.lsh_attn.rotations.weight
            torch.save(weight, prefix + '%d.pt' % i)


class ReformerLM(nn.Module):
    def __init__(self, num_tokens, dim, depth, max_seq_len, heads=8, dim_head=None, bucket_size=64, n_hashes=4,
                 ff_chunks=100, attn_chunks=1, causal=False, weight_tie=False, lsh_dropout=0., ff_dropout=0., ff_mult=4,
                 ff_activation=None, ff_glu=False, post_attn_dropout=0., layer_dropout=0.,
                 random_rotations_per_head=False, twin_attention=False, use_scale_norm=False, use_rezero=False,
                 use_full_attn=False, full_attn_thres=0, reverse_thres=0, num_mem_kv=0, one_value_head=False,
                 emb_dim=None, return_embeddings=False, weight_tie_embedding=False, fixed_position_emb=False,
                 absolute_position_emb=False, axial_position_shape=None, n_local_attn_heads=0, pkm_layers=tuple(),
                 pkm_num_keys=128, attn_type='lsh', store_stats=False):
        super().__init__()
        emb_dim = default(emb_dim, dim)
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(num_tokens, emb_dim)

        self.to_model_dim = Identity() if emb_dim == dim else nn.Linear(emb_dim, dim)

        if absolute_position_emb:
            # self.pos_emb = PositionalEncoding(emb_dim, layer_dropout, max_seq_len)
            self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len)
        elif fixed_position_emb:
            self.pos_emb = FixedPositionalEmbedding(emb_dim)
        else:
            axial_position_shape = default(axial_position_shape, (max_seq_len // bucket_size, bucket_size))
            self.pos_emb = AxialPositionalEmbedding(emb_dim, axial_position_shape)

        self.reformer = Reformer(dim, depth, max_seq_len, heads=heads, dim_head=dim_head, bucket_size=bucket_size,
                                 n_hashes=n_hashes, ff_chunks=ff_chunks, attn_chunks=attn_chunks, causal=causal,
                                 weight_tie=weight_tie, lsh_dropout=lsh_dropout, ff_mult=ff_mult,
                                 ff_activation=ff_activation, ff_glu=ff_glu, ff_dropout=ff_dropout,
                                 post_attn_dropout=0., layer_dropout=layer_dropout,
                                 random_rotations_per_head=random_rotations_per_head, twin_attention=twin_attention,
                                 use_scale_norm=use_scale_norm, use_rezero=use_rezero, use_full_attn=use_full_attn,
                                 full_attn_thres=full_attn_thres, reverse_thres=reverse_thres, num_mem_kv=num_mem_kv,
                                 one_value_head=one_value_head, n_local_attn_heads=n_local_attn_heads,
                                 pkm_layers=pkm_layers, pkm_num_keys=pkm_num_keys, attn_type=attn_type,
                                 store_stats=store_stats)

        if return_embeddings:
            self.out = Identity()
            return

        self.out = nn.Sequential(
            nn.Linear(dim, emb_dim) if emb_dim != dim else Identity(),
            nn.Linear(emb_dim, num_tokens) if not weight_tie_embedding else MatrixMultiply(self.token_emb.weight,
                                                                                           transpose=True,
                                                                                           normalize=True)
        )
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.token_emb.weight.data.uniform_(-initrange, initrange)
        self.out[1].bias.data.zero_()
        self.out[1].weight.data.uniform_(-initrange, initrange)

    def forward(self, x, **kwargs):
        x = self.token_emb(x)
        x = x + self.pos_emb(x).type_as(x)
        # x = self.pos_emb(x)

        x = self.to_model_dim(x)
        x = self.reformer(x, **kwargs)
        return self.out(x)

    def clear_non_rotation_gradients(self):
        # clear gradients from triplet loss
        for i in range(len(self.reformer.layer_modules) // 2):
            f = self.reformer.layer_modules[2 * i].fn
            g = self.reformer.layer_modules[(2 * i) + 1].fn
            # only zero out toqk, tov, and to_out
            # leave gradients of rotations
            f.toqk.zero_grad()
            f.tov.zero_grad()
            f.to_out.zero_grad()
            g.zero_grad()

    def get_triplet_loss(self):
        total = 0
        for i in range(len(self.reformer.layer_modules) // 2):
            f = self.reformer.layer_modules[2 * i].fn
            if f.triplet_loss is not None:
                total += f.triplet_loss
        return total

    def update_simhash(self):
        for i in range(len(self.reformer.layer_modules) // 2):
            f = self.reformer.layer_modules[2 * i].fn
            attn_fn = f.lsh_attn
            if hasattr(attn_fn, 'simhash'):
                attn_fn.update_simhash()

    def reset_triplet(self):
        for i in range(len(self.reformer.layer_modules) // 2):
            f = self.reformer.layer_modules[2 * i].fn
            attn_fn = f.lsh_attn
            if hasattr(attn_fn, 'rotations'):
                attn_fn.reset_rotations()

    def set_alpha(self, alpha):
        for i in range(len(self.reformer.layer_modules) // 2):
            f = self.reformer.layer_modules[2 * i].fn
            attn_fn = f.lsh_attn
            if hasattr(attn_fn, 'alpha'):
                attn_fn.alpha = alpha

    def get_statistics(self, batch_size):
        means = []
        for i in range(len(self.reformer.layer_modules) // 2):
            f = self.reformer.layer_modules[2 * i].fn
            attn_fn = f.lsh_attn
            means.append(attn_fn.mean_dp / attn_fn.stat_count)
            attn_fn.mean_dp = 0.0
            attn_fn.stat_count = 0
        return means

    def clear_triplet_loss(self):
        for i in range(len(self.reformer.layer_modules) // 2):
            f = self.reformer.layer_modules[2 * i].fn
            if f.triplet_loss is not None:
                f.triplet_loss = None

    def save_triplet_params(self, prefix):
        for i in range(len(self.reformer.layer_modules) // 2):
            f = self.reformer.layer_modules[2 * i].fn
            weight = f.lsh_attn.rotations.weight
            torch.save(weight, prefix + '%d.pt' % i)
