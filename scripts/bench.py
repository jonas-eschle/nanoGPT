#!/usr/bin/env python
"""
Script for benchmarking a GPT model
"""

import os
import sys
import time
from contextlib import nullcontext

# Add the parent directory to the path so we can import nanogpt


import numpy as np
import torch

from nanogpt.config import Config
from nanogpt.model import GPT, GPTConfig


def main():
    # Create default configuration
    config = Config(
        batch_size=12,
        block_size=1024,
        bias=False,
        real_data=True,
        seed=1337,
        device="cpu",
        dtype="bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16",
        compile=True,
        profile=False,
    )

    # Override with command line arguments or config file
    config = Config.from_command_line(config)

    # Setup
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in config.device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
        config.dtype
    ]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    # Data loading init
    if config.real_data:
        dataset = "openwebtext"
        data_dir = os.path.join("data", dataset)
        train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")

        def get_batch(split):
            data = train_data  # note ignore split in benchmarking script
            ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
            x = torch.stack(
                [torch.from_numpy((data[i : i + config.block_size]).astype(np.int64)) for i in ix]
            )
            y = torch.stack(
                [
                    torch.from_numpy((data[i + 1 : i + 1 + config.block_size]).astype(np.int64))
                    for i in ix
                ]
            )
            x, y = (
                x.pin_memory().to(config.device, non_blocking=True),
                y.pin_memory().to(config.device, non_blocking=True),
            )
            return x, y

    else:
        # alternatively, if fixed data is desired to not care about data loading
        x = torch.randint(50304, (config.batch_size, config.block_size), device=config.device)
        y = torch.randint(50304, (config.batch_size, config.block_size), device=config.device)
        get_batch = lambda split: (x, y)

    # Model init
    gptconf = GPTConfig(
        block_size=config.block_size,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0,
        bias=config.bias,
    )
    model = GPT(gptconf)
    model.to(config.device)

    optimizer = model.configure_optimizers(
        weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device_type
    )

    if config.compile:
        print("Compiling model...")
        model = torch.compile(model)  # pytorch 2.0

    if config.profile:
        # Profiling with PyTorch profiler
        wait, warmup, active = 5, 5, 5
        num_steps = wait + warmup + active
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./bench_log"),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=True,
            with_modules=False,
        ) as prof:
            X, Y = get_batch("train")
            for k in range(num_steps):
                with ctx:
                    logits, loss = model(X, Y)
                X, Y = get_batch("train")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                lossf = loss.item()
                print(f"{k}/{num_steps} loss: {lossf:.4f}")
                prof.step()
    else:
        # Simple benchmarking
        torch.cuda.synchronize()
        for stage, num_steps in enumerate([10, 20]):  # burnin, then benchmark
            t0 = time.time()
            X, Y = get_batch("train")
            for k in range(num_steps):
                with ctx:
                    logits, loss = model(X, Y)
                X, Y = get_batch("train")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                lossf = loss.item()
                print(f"{k}/{num_steps} loss: {lossf:.4f}")
            torch.cuda.synchronize()
            t1 = time.time()
            dt = t1 - t0
            mfu = model.estimate_mfu(config.batch_size * 1 * num_steps, dt)
            if stage == 1:
                print(f"time per iteration: {dt/num_steps*1000:.4f}ms, MFU: {mfu*100:.2f}%")


if __name__ == "__main__":
    main()
