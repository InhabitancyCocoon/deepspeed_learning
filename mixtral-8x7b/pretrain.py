import argparse
import functools
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.profiler as profiler
import pyarrow.parquet as pq
import os
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from datetime import datetime
from torch.utils.data import DataLoader, Dataset, get_worker_info
from torch.optim.lr_scheduler import StepLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    transformer_auto_wrap_policy,
    wrap,
)
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    MixtralForCausalLM,
    AutoConfig,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralDecoderLayer,
    MixtralAttention,
    MixtralSparseMoeBlock
)

MAX_SEQ_LENGTH = 1024

import deepspeed
import torch.distributed as dist
from deepspeed.ops.adam import FusedAdam
from deepspeed import get_accelerator


def set_parameter_bf16(model: nn.Module):
    for param in model.parameters():
        param.data = param.data.to(torch.bfloat16)

def mixtral_tokenizer_collate_fn(batch):
    x, y = [], []
    for item in batch:
        x.append(item[0]['input_ids'])
        y.append(item[1]['input_ids'])
    x = torch.cat(x)
    y = torch.cat(y)
    return x, y



def print_trainable_parameters(model):
    """
    in the mlp block, there seems to be a gate proj layer combined with up proj layer, same as mistral 7b.
    model.layers.0.self_attn.q_proj.weight ([14336, 4096])
    the size of k, v attention matrices are torch.Size([1024, 4096]), a little strange, it may be related to flash attention or sdpa ?
    the parameters is saved in torch.bf16, but run in torch.fp32
    roughly 46B
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        # print(name, param.data.shape, param.data.dtype)
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


class MixtralDatset(Dataset):
    def __init__(self) -> None:
        model_path = './fake_model'
        self.file = pq.read_table('./data/pretrain_lora.parquet', columns=['text']).to_pandas()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return self.file.shape[0]

    def __getitem__(self, index):
        row = self.file.iloc[index].values[0]
        x = self.tokenizer(row, return_tensors='pt', max_length=MAX_SEQ_LENGTH, truncation=True, padding='max_length')
        return x, x


def train(model, data, target, optimizer, lr_scheduler, rank, step):
    model.train()
    data, target = data.to(rank), target.to(rank)
    output = model(data, use_cache=False)
    loss = F.cross_entropy(output.logits.view(-1, output.logits.size(-1)), target.view(-1))
    

    # <class 'deepspeed.runtime.engine.DeepSpeedEngine'> 
    # <class 'deepspeed.runtime.zero.stage3.DeepSpeedZeroOptimizer_Stage3'>
    # print(type(model), type(optimizer))

    model.backward(loss)
    model.step()

    ds_loss = torch.zeros(2).to(rank)
    ds_loss[0] += loss.item()
    ds_loss[1] += len(data)

    dist.all_reduce(ds_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(f'Train Step: {step}, Loss: {ds_loss[0] / ds_loss[1]: .3f}')
    


def ds_main(args):
    rank = args.local_rank
    torch.cuda.set_device(rank)
    deepspeed.init_distributed(dist_backend="nccl")


    config = AutoConfig.from_pretrained('./fake_model/config.json')
    model_path = './fake_model'
    with init_empty_weights():
        config = AutoConfig.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        # set use_cache = False:
        config.use_cache = False
        model = MixtralForCausalLM(config)

    set_parameter_bf16(model)
    model.to_empty(device=torch.cuda.current_device())

    if rank == 0:
        # model.from_pretrained(model_path)
        print_trainable_parameters(model)

    if args.gradient_checkpointing:
        gradient_checkpointing_kwargs = {'use_reentrant': False}
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
 
    

    train_dataset = MixtralDatset()
    train_sampler = DistributedSampler(train_dataset, rank=rank)
    train_kwargs = {
        'batch_size': args.per_device_batch_size,
        'sampler': train_sampler,
        'num_workers': 2,
        'pin_memory': True,
        'collate_fn': mixtral_tokenizer_collate_fn,
        'pin_memory': True,
        'pin_memory_device': f'cuda:{rank}'
    }
    train_loader = DataLoader(train_dataset, **train_kwargs)

    optimizer = FusedAdam(model.parameters(), lr=args.lr)

    lr_scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    ds_config = {
        "train_batch_size": args.global_batch_size,
        "train_micro_batch_size_per_gpu": args.per_device_batch_size,
        "steps_per_print": 10,
        "zero_optimization": {
            "stage": args.stage,
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dist_init_required=True,
        config=ds_config,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    step = 0
    for data, target in train_loader:
        train(model, data, target, optimizer, lr_scheduler, rank, step)
        step += 1
        if step >= args.steps:
            break




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepSpeed Example')

    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--per-device-batch-size", type=int, default=8)

    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--master_port", type=int)

    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--gamma", type=float, default=0.75)

    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--step-size", type=int, default=2)
    parser.add_argument("--gradient-checkpointing", type=bool, default=True)

    parser.add_argument("--seed", type=int, default=42)


    args = parser.parse_args()
    torch.manual_seed(args.seed)
    ds_main(args)