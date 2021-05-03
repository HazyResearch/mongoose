import random
import math
import numpy as np
import torch
import argparse
import time
import os

from reformer_lib.reformer_pytorch import ReformerLM,ReformerLM_tune
from reformer_lib.generative_tools import TrainingWrapper
from scheduler import Scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    # from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("This code requires APEX")

seed=17
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="synthetic")
parser.add_argument('--seq_len', type=int, default=1024)
parser.add_argument('--min_seq_len', type=int, default=4)
parser.add_argument('--ntokens', type=int, default=16)
parser.add_argument('--emsize', type=int, default=256)
parser.add_argument('--nhid', type=int, default=256)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--nhead', type=int, default=4)
parser.add_argument('--bucket_size_list', nargs='+', type=int, default=[64, 64])
parser.add_argument('--n_hashes_list', nargs='+', type=int, default=[1, 1])
parser.add_argument('--attn_type_list', nargs='+', default=['lsh', 'lsh'])
parser.add_argument('--dropout', type=float, default=0.05)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--full_attn_thres', type=int, default=0)
parser.add_argument('--lr_main', type=float, default=1e-3)
parser.add_argument('--lr_tri', type=float, default=1e-3)
parser.add_argument('--tri_alpha', type=float, default=1.0)
parser.add_argument('--use_full_attn', action='store_true')
parser.add_argument('--log', action='store_false')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--train_batches', type=int, default=5000)
parser.add_argument('--eval_batches', type=int, default=500)
parser.add_argument('--print_loss', type=int, default=500)
parser.add_argument('--note', type=str, default='')
parser.add_argument('--K', type=int, default=1)
parser.add_argument('--L', type=int, default=10)
parser.add_argument('--local_rank', type=int, default=0)


GRADIENT_ACCUMULATE_EVERY = 1
args = parser.parse_args()

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

args.gpu = 0
args.world_size = 1

if args.distributed:
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    gpu_id = rank % num_gpus
    torch.cuda.set_device(gpu_id)
    # args.gpu = args.local_rank
    # torch.cuda.set_device(args.gpu)
    torch.distributed.init_process_group(backend='nccl')
    args.world_size = torch.distributed.get_world_size()

# generating Tensorboard writer
if args.log and args.local_rank==0:
    log_dir = "./log_paper/{}".format(args.dataset)
    log_file = log_dir + "/{}_bucket{}_hash{}_seq{}_bz{}_token{}_lr{}_alpha{}_layer{}_note{}.txt".format(
        '_'.join(args.attn_type_list), '_'.join(str(x) for x in args.bucket_size_list), '_'.join(str(x) for x in args.n_hashes_list),
        args.seq_len, args.batch_size, args.ntokens, args.lr_tri, args.tri_alpha, args.nlayers, args.note)
    os.makedirs(log_dir, exist_ok=True)
    print("args: ", args, file=open(log_file, "a"))
    print("args: ", args)
    run_file = "./run_paper/{}/{}_bucket{}_hash{}_seq{}_bz{}_token{}_lr{}_alpha{}_layer{}_note{}".format(args.dataset,
                                                  '_'.join(args.attn_type_list),'_'.join(str(x) for x in args.bucket_size_list),'_'.join(str(x) for x in args.n_hashes_list),
                                                  args.seq_len, args.batch_size,
                                                  args.ntokens, args.lr_tri, args.tri_alpha,
                                                  args.nlayers, args.note)
    writer = SummaryWriter(run_file)


def save_model(model, optimizer, name, iteration):
    with open(os.path.join('synthetic', 'model_' + name + '_' + str(iteration) + '.pt'), 'wb') as f:
        torch.save(model.state_dict(), f)
    with open(os.path.join('synthetic', 'optimizer_' + name + '_' + str(iteration) + '.pt'), 'wb') as f:
        torch.save(optimizer.state_dict(), f)


def _pad_to_multiple_of(x, y, axis):
    """Pads x to multiple of y on the given axis."""
    pad_len = np.ceil(x.shape[axis] / float(y)) * y
    pad_widths = [(0, 0)] * len(x.shape)
    pad_widths[axis] = (0, int(pad_len - x.shape[axis]))
    return np.pad(x, pad_widths, mode='constant',
                  constant_values=x.dtype.type(0))


def sequence_copy_inputs(
        vocab_size, batch_size, train_length,
        eval_min_length, eval_max_length, reverse=False,
        pad_to_multiple=32):
    """Inputs for the sequence copy problem: 0w0w for w in [1..vocab_size-1]*.
  Args:
    vocab_size: how many symbols to use.
    batch_size: how large are the batches.
    train_length: maximum length of w for training.
    eval_min_length: minimum length of w for eval.
    eval_max_length : maximum length of w for eval.
    reverse: bool (optional, false by default): reverse the second sequence.
    pad_to_multiple: int, pad length to be multiple of this number.
  Returns:
    trax.inputs.Inputs
  """

    def random_minibatches(length_list):
        """Generate a stream of random mini-batches."""
        while True:
            length = random.choice(length_list)
            assert length % 2 == 0
            w_length = (length // 2) - 1
            w = np.random.randint(low=1, high=vocab_size - 1,
                                  size=(batch_size, w_length))
            zero = np.zeros([batch_size, 1], np.int32)
            loss_weights = np.concatenate([np.zeros((batch_size, w_length + 2)),
                                           np.ones((batch_size, w_length))], axis=1)
            if reverse:
                x = np.concatenate([zero, w, zero, np.flip(w, axis=1)], axis=1)
            else:
                x = np.concatenate([zero, w, zero, w], axis=1)
            x = torch.Tensor(_pad_to_multiple_of(x, pad_to_multiple, 1)).cuda().long()
            loss_weights = torch.Tensor(_pad_to_multiple_of(loss_weights, pad_to_multiple, 1)).cuda().long()
            yield (x,x,loss_weights)  # Here inputs and targets are the same.

    train_lengths = [2 * (i + 2) for i in range(train_length - 1)]
    eval_lengths = [2 * (i + 1) for i in range(eval_min_length, eval_max_length)]
    train_stream = lambda _: random_minibatches(train_lengths)
    eval_stream = lambda _: random_minibatches(eval_lengths)
    return train_stream(None), eval_stream(None)


def train(model, train_loader, epoch):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    for batch in range(args.train_batches):
        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            data, target, loss_mask = next(train_loader)
            if 'triplet' in args.attn_type_list:
                loss = model(data, loss_weight=loss_mask, return_loss=True, calc_triplet=True)
                tri_loss = model.net.net.get_triplet_loss()
                if tri_loss != 0.0:
                    tri_loss.backward()
                    model.net.net.clear_triplet_loss()
            else:
                loss = model(data, loss_weight=loss_mask, return_loss=True, calc_triplet=False)
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer_main.step()
        optimizer_main.zero_grad()
        if 'triplet' in args.attn_type_list:
            optimizer_tri.step()
            optimizer_tri.zero_grad()

        total_loss += loss.item()
        log_interval = args.print_loss
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            if args.local_rank==0:
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                      'lr {:02.5f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, args.train_batches , args.lr_main,
                                  elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            if args.log and args.local_rank==0:
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                      'lr {:02.5f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, args.train_batches , args.lr_main,
                                  elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)), file = open(log_file, "a"))
            total_loss = 0
            if args.log and args.local_rank==0:
                writer.add_scalar('Loss/train', cur_loss, epoch*args.train_batches+batch)
                writer.add_scalar('Loss/train_pp', math.exp(cur_loss), epoch*args.train_batches+batch)
            start_time = time.time()


def evaluate(eval_model, data_source, data_batches, epoch):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    counter = 0
    with torch.no_grad():
        for i in range(data_batches):
            data, target, loss_mask = next(data_source)
            loss = eval_model(data, loss_weight=loss_mask, return_loss=True,
                              calc_triplet=False)
            valid_token = torch.sum(loss_mask)
            total_loss += valid_token*loss.item()
            counter += valid_token
    return total_loss / counter


if __name__ == "__main__":
    train_loader, eval_loader = sequence_copy_inputs(args.ntokens, args.batch_size,
                                                     args.seq_len//2, args.min_seq_len,
                                                     args.seq_len//2)

    model = ReformerLM_tune(
        dim=args.nhid,
        emb_dim=args.emsize,
        depth=args.nlayers,
        max_seq_len=args.seq_len,
        num_tokens=args.ntokens,
        heads=args.nhead,
        bucket_size_list=args.bucket_size_list,
        fixed_position_emb=True,
        n_hashes_list=args.n_hashes_list,
        ff_chunks=1,
        ff_mult=1,
        attn_chunks=1,
        layer_dropout=0.,
        ff_dropout=args.dropout,
        post_attn_dropout=args.dropout,
        lsh_dropout=args.dropout,
        weight_tie=False,
        causal=True,
        n_local_attn_heads=0,
        use_full_attn=args.use_full_attn,  # set this to true for comparison with full attention
        reverse_thres=9999999999,
        full_attn_thres=args.full_attn_thres,
        num_mem_kv=0,
        attn_type_list=args.attn_type_list,
        store_stats=args.log,
        pkm_num_keys=0,
    )

    model = TrainingWrapper(model)
    model.cuda()

    params_1 = []
    params_2 = []
    for name, p in model.named_parameters():
        if 'rotation' in name:
            params_2.append(p)
        else:
            params_1.append(p)

    if 'triplet' in args.attn_type_list:
        optimizer_main = torch.optim.Adam(params_1, lr=args.lr_main)
        optimizer_tri = torch.optim.Adam(params_2, lr=args.lr_tri)
    else:
        optimizer_main = torch.optim.Adam(model.parameters(), lr=args.lr_main)

    if args.distributed:
        global ddp_model
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
        ddp_model = model

    for epoch in range(0, args.epochs):
        epoch_start_time = time.time()
        train(model, train_loader, epoch)
        val_loss = evaluate(model, eval_loader, args.print_loss, epoch)

        if args.log and args.local_rank==0:
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Loss/val_pp', math.exp(val_loss), epoch)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
        if args.log and args.local_rank==0:
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss)), file=open(log_file, "a"))

    if args.log:
        writer.close()
