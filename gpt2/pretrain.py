import argparse
from ast import arg
from dis import dis
import math
import functools
import os
import logging
import torch.nn.functional as F
import torch
import pyarrow.parquet as pq
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoConfig
from torch.optim.lr_scheduler import StepLR
import deepspeed
import torch.distributed as dist
from deepspeed.ops.adam import FusedAdam
from deepspeed import get_accelerator



class GPT2Datset(Dataset):
    def __init__(self) -> None:
        self.file = pq.read_table('./data/pretrain_lora.parquet', columns=['text']).to_pandas()
        self.tokenizer = GPT2Tokenizer.from_pretrained('./fake_model')
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return self.file.shape[0]

    def __getitem__(self, index):
        row = self.file.iloc[index].values[0]
        x = self.tokenizer(row, return_tensors='pt', max_length=1024, truncation=True, padding='max_length')['input_ids']
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
    config = AutoConfig.from_pretrained('./fake_model/config.json')
    model = GPT2LMHeadModel._from_config(config)
    deepspeed.init_distributed(dist_backend="nccl")

    train_dataset = GPT2Datset()
    train_sampler = DistributedSampler(train_dataset, rank=rank)
    train_kwargs = {
        'batch_size': args.per_device_batch_size,
        'sampler': train_sampler,
        'num_workers': 2,
        'pin_memory': True
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


"""
Train Step: 0, Loss:  1.315
Train Step: 1, Loss:  0.980
Train Step: 2, Loss:  0.879
Train Step: 3, Loss:  0.815
Train Step: 4, Loss:  0.812
Train Step: 5, Loss:  0.898
Train Step: 6, Loss:  0.756
Train Step: 7, Loss:  0.737
Train Step: 8, Loss:  0.672
"""