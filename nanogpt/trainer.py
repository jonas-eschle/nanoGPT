"""
Training functionality for the GPT model.
"""

import math
import os
import pickle
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from nanogpt.model import GPT, GPTConfig


class Trainer:
    """
    Trainer class for GPT models
    """

    def __init__(self, config=None):
        self.config = {} if config is None else config
        # Default config values
        self.out_dir = self.config.get("out_dir", "out")
        self.eval_interval = self.config.get("eval_interval", 2000)
        self.log_interval = self.config.get("log_interval", 1)
        self.eval_iters = self.config.get("eval_iters", 200)
        self.eval_only = self.config.get("eval_only", False)
        self.always_save_checkpoint = self.config.get("always_save_checkpoint", True)
        self.init_from = self.config.get("init_from", "scratch")

        # wandb logging
        self.wandb_log = self.config.get("wandb_log", False)
        self.wandb_project = self.config.get("wandb_project", "owt")
        self.wandb_run_name = self.config.get("wandb_run_name", "gpt2")

        # data
        self.dataset = self.config.get("dataset", "openwebtext")
        self.gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 5 * 8)
        self.batch_size = self.config.get("batch_size", 12)
        self.block_size = self.config.get("block_size", 1024)

        # model
        self.n_layer = self.config.get("n_layer", 12)
        self.n_head = self.config.get("n_head", 12)
        self.n_embd = self.config.get("n_embd", 768)
        self.dropout = self.config.get("dropout", 0.0)
        self.bias = self.config.get("bias", False)

        # optimizer
        self.learning_rate = self.config.get("learning_rate", 6e-4)
        self.max_iters = self.config.get("max_iters", 600000)
        self.weight_decay = self.config.get("weight_decay", 1e-1)
        self.beta1 = self.config.get("beta1", 0.9)
        self.beta2 = self.config.get("beta2", 0.95)
        self.grad_clip = self.config.get("grad_clip", 1.0)

        # learning rate decay
        self.decay_lr = self.config.get("decay_lr", True)
        self.warmup_iters = self.config.get("warmup_iters", 2000)
        self.lr_decay_iters = self.config.get("lr_decay_iters", 600000)
        self.min_lr = self.config.get("min_lr", 6e-5)

        # DDP settings
        self.backend = self.config.get("backend", "nccl")

        # system
        self.device = self.config.get("device", "cuda")
        self.dtype = self.config.get(
            "dtype",
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16",
        )
        self.compile = self.config.get("compile", True)

        # setup
        self.ddp = int(os.environ.get("RANK", -1)) != -1
        if self.ddp:
            init_process_group(backend=self.backend)
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.device = f"cpu:{self.ddp_local_rank}"
            torch.set_default_device(self.device)
            self.master_process = self.ddp_rank == 0
            self.seed_offset = self.ddp_rank
            assert self.gradient_accumulation_steps % self.ddp_world_size == 0
            self.gradient_accumulation_steps //= self.ddp_world_size
        else:
            self.master_process = True
            self.seed_offset = 0
            self.ddp_world_size = 1

        self.tokens_per_iter = (
            self.gradient_accumulation_steps
            * self.ddp_world_size
            * self.batch_size
            * self.block_size
        )

        if self.master_process:
            os.makedirs(self.out_dir, exist_ok=True)

        torch.manual_seed(1337 + self.seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.device_type = "cuda" if "cuda" in self.device else "cpu"
        self.ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.dtype]
        self.ctx = (
            nullcontext()
            if self.device_type == "cpu"
            else torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)
        )

        # model setup
        self.iter_num = 0
        self.best_val_loss = 1e9
        self.model = None
        self.optimizer = None
        self.scaler = None

    def get_batch(self, split):
        """
        Get a batch of data from the dataset
        """
        data_dir = Path(__file__).absolute().parent.parent /  "data" / self.dataset
        # We recreate np.memmap every batch to avoid a memory leak
        if split == "train":
            data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
        else:
            data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack(
            [torch.from_numpy((data[i : i + self.block_size]).astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [torch.from_numpy((data[i + 1 : i + 1 + self.block_size]).astype(np.int64)) for i in ix]
        )
        if self.device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = (
                x.pin_memory().to(self.device, non_blocking=True),
                y.pin_memory().to(self.device, non_blocking=True),
            )
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    def init_model(self):
        """
        Initialize the model
        """
        # attempt to derive vocab_size from the dataset
        data_dir = os.path.join("data", self.dataset)
        meta_path = os.path.join(data_dir, "meta.pkl")
        meta_vocab_size = None
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            meta_vocab_size = meta["vocab_size"]
            if self.master_process:
                print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

        # model init
        model_args = dict(
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            block_size=self.block_size,
            bias=self.bias,
            vocab_size=None,
            dropout=self.dropout,
        )

        if self.init_from == "scratch":
            # init a new model from scratch
            if self.master_process:
                print("Initializing a new model from scratch")
            # determine the vocab size we'll use for from-scratch training
            if meta_vocab_size is None:
                if self.master_process:
                    print(
                        "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
                    )
            model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
            gptconf = GPTConfig(**model_args)
            self.model = GPT(gptconf)
        elif self.init_from == "resume":
            if self.master_process:
                print(f"Resuming training from {self.out_dir}")
            # resume training from a checkpoint.
            ckpt_path = os.path.join(self.out_dir, "ckpt.pt")
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            checkpoint_model_args = checkpoint["model_args"]
            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
                model_args[k] = checkpoint_model_args[k]
            # create the model
            gptconf = GPTConfig(**model_args)
            self.model = GPT(gptconf)
            state_dict = checkpoint["model"]
            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            unwanted_prefix = "_orig_mod."
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
            self.model.load_state_dict(state_dict)
            self.iter_num = checkpoint["iter_num"]
            self.best_val_loss = checkpoint["best_val_loss"]
        elif self.init_from.startswith("gpt2"):
            if self.master_process:
                print(f"Initializing from OpenAI GPT-2 weights: {self.init_from}")
            # initialize from OpenAI GPT-2 weights
            override_args = dict(dropout=self.dropout)
            self.model = GPT.from_pretrained(self.init_from, override_args)
            # read off the created config params, so we can store them into checkpoint correctly
            for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
                model_args[k] = getattr(self.model.config, k)

        # crop down the model block size if desired, using model surgery
        if self.block_size < self.model.config.block_size:
            self.model.crop_block_size(self.block_size)
            model_args[
                "block_size"
            ] = self.block_size  # so that the checkpoint will have the right value

        self.model.to(self.device)

        # initialize a GradScaler. If enabled=False scaler is a no-op
        self.scaler = torch.amp.GradScaler('cpu', enabled=(self.dtype == "float16"))

        # optimizer
        self.optimizer = self.model.configure_optimizers(
            self.weight_decay, self.learning_rate, (self.beta1, self.beta2), self.device_type
        )
        if self.init_from == "resume":
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        # compile the model
        if self.compile:
            if self.master_process:
                print("compiling the model... (takes a ~minute)")
            self.unoptimized_model = self.model
            self.model = torch.compile(self.model)  # requires PyTorch 2.0

        # wrap model into DDP container
        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])

        return self.model

    @torch.no_grad()
    def estimate_loss(self):
        """
        Estimate loss over train and validation splits
        """
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.get_batch(split)
                with self.ctx:
                    logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def get_lr(self, it):
        """
        Learning rate decay scheduler (cosine with warmup)
        """
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return self.learning_rate * (it + 1) / (self.warmup_iters + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)

    def train(self):
        """
        Main training loop
        """
        if self.model is None:
            self.init_model()

        # logging
        if self.wandb_log and self.master_process:
            import wandb

            wandb.init(project=self.wandb_project, name=self.wandb_run_name, config=self.config)

        # training loop
        X, Y = self.get_batch("train")  # fetch the very first batch
        t0 = time.time()
        local_iter_num = 0  # number of iterations in the lifetime of this process
        raw_model = self.model.module if self.ddp else self.model  # unwrap DDP container if needed
        running_mfu = -1.0

        while True:
            # determine and set the learning rate for this iteration
            lr = self.get_lr(self.iter_num) if self.decay_lr else self.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if self.iter_num % self.eval_interval == 0 and self.master_process:
                losses = self.estimate_loss()
                print(
                    f"step {self.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )
                if self.wandb_log:
                    wandb.log(
                        {
                            "iter": self.iter_num,
                            "train/loss": losses["train"],
                            "val/loss": losses["val"],
                            "lr": lr,
                            "mfu": running_mfu * 100,  # convert to percentage
                        }
                    )
                if losses["val"] < self.best_val_loss or self.always_save_checkpoint:
                    self.best_val_loss = losses["val"]
                    if self.iter_num > 0:
                        checkpoint = {
                            "model": raw_model.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "model_args": model_args,
                            "iter_num": self.iter_num,
                            "best_val_loss": self.best_val_loss,
                            "config": self.config,
                        }
                        print(f"saving checkpoint to {self.out_dir}")
                        torch.save(checkpoint, os.path.join(self.out_dir, "ckpt.pt"))
            if self.iter_num == 0 and self.eval_only:
                break

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(self.gradient_accumulation_steps):
                if self.ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    self.model.require_backward_grad_sync = (
                        micro_step == self.gradient_accumulation_steps - 1
                    )
                with self.ctx:
                    logits, loss = self.model(X, Y)
                    loss = (
                        loss / self.gradient_accumulation_steps
                    )  # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = self.get_batch("train")
                # backward pass, with gradient scaling if training in fp16
                self.scaler.scale(loss).backward()
            # clip the gradient
            if self.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            # step the optimizer and scaler if training in fp16
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            self.optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if self.iter_num % self.log_interval == 0 and self.master_process:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * self.gradient_accumulation_steps
                if local_iter_num >= 5:  # let the training loop settle a bit
                    mfu = raw_model.estimate_mfu(
                        self.batch_size * self.gradient_accumulation_steps, dt
                    )
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                print(
                    f"iter {self.iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
                )
            self.iter_num += 1
            local_iter_num += 1

            # termination conditions
            if self.iter_num > self.max_iters:
                break

        if self.ddp:
            destroy_process_group()
