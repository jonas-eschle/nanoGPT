#!/usr/bin/env python
"""
Script for training a GPT model

This script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python scripts/train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 scripts/train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 scripts/train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 scripts/train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import sys

# disable GPU for torch
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


from nanogpt.config import Config
from nanogpt.trainer import Trainer


def main():
    # Create default configuration
    config = Config(
        # I/O
        out_dir="out",
        eval_interval=2000,
        log_interval=1,
        eval_iters=200,
        eval_only=False,
        always_save_checkpoint=True,
        init_from="scratch",
        # wandb logging
        wandb_log=False,
        wandb_project="owt",
        wandb_run_name="gpt2",
        # data
        dataset="shakespeare",
        gradient_accumulation_steps=5 * 8,
        batch_size=12,
        block_size=1024,
        # model
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=False,
        # optimizer
        learning_rate=6e-4,
        max_iters=600000,
        weight_decay=1e-1,
        beta1=0.9,
        beta2=0.95,
        grad_clip=1.0,
        # learning rate decay
        decay_lr=True,
        warmup_iters=2,
        lr_decay_iters=600000,
        min_lr=6e-5,
        # DDP settings
        backend="gloo",
        # system
        device="cpu",
        dtype="bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16",
        compile=True,
    )

    # Override with command line arguments or config file
    config = Config.from_command_line(config)

    # Create trainer and train
    trainer = Trainer(config.to_dict())
    trainer.train()


if __name__ == "__main__":
    import torch

    main()
