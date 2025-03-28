"""
Sampling functionality for the GPT model.
"""

import os
import pickle
from contextlib import nullcontext

import tiktoken
import torch

from nanogpt.model import GPT, GPTConfig


class Sampler:
    """
    Sampler class for generating text from a GPT model
    """

    def __init__(self, config=None):
        self.config = {} if config is None else config
        # Default config values
        self.init_from = self.config.get("init_from", "resume")
        self.out_dir = self.config.get("out_dir", "out")
        self.start = self.config.get("start", "\n")
        self.num_samples = self.config.get("num_samples", 10)
        self.max_new_tokens = self.config.get("max_new_tokens", 500)
        self.temperature = self.config.get("temperature", 0.8)
        self.top_k = self.config.get("top_k", 200)
        self.seed = self.config.get("seed", 1337)
        self.device = self.config.get("device", "cpu")
        self.dtype = self.config.get(
            "dtype",
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16",
        )
        self.compile = self.config.get("compile", False)

        # setup
        torch.manual_seed(self.seed)
        # torch.cuda.manual_seed(self.seed)
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

        self.model = None
        self.encode = None
        self.decode = None

    def init_model(self):
        """
        Initialize the model
        """
        if self.init_from == "resume":
            # init from a model saved in a specific directory
            ckpt_path = os.path.join(self.out_dir, "ckpt.pt")
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            gptconf = GPTConfig(**checkpoint["model_args"])
            self.model = GPT(gptconf)
            state_dict = checkpoint["model"]
            unwanted_prefix = "_orig_mod."
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
            self.model.load_state_dict(state_dict)
        elif self.init_from.startswith("gpt2"):
            # init from a given GPT-2 model
            self.model = GPT.from_pretrained(self.init_from, dict(dropout=0.0))

        self.model.eval()
        self.model.to(self.device)
        if self.compile:
            self.model = torch.compile(self.model)  # requires PyTorch 2.0 (optional)

        # setup tokenizer
        self.setup_tokenizer(checkpoint if self.init_from == "resume" else None)

        return self.model

    def setup_tokenizer(self, checkpoint=None):
        """
        Setup tokenizer for encoding/decoding text
        """
        # look for the meta pickle in case it is available in the dataset folder
        load_meta = False
        if checkpoint is not None and "config" in checkpoint and "dataset" in checkpoint["config"]:
            meta_path = os.path.join("data", checkpoint["config"]["dataset"], "meta.pkl")
            load_meta = os.path.exists(meta_path)

        if load_meta:
            print(f"Loading meta from {meta_path}...")
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            # TODO want to make this more general to arbitrary encoder/decoder schemes
            stoi, itos = meta["stoi"], meta["itos"]
            self.encode = lambda s: [stoi[c] for c in s]
            self.decode = lambda l: "".join([itos[i] for i in l])
        else:
            # ok let's assume gpt-2 encodings by default
            print("No meta.pkl found, assuming GPT-2 encodings...")
            enc = tiktoken.get_encoding("gpt2")
            self.encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            self.decode = lambda l: enc.decode(l)

    def generate(self, prompt=None):
        """
        Generate text samples from the model
        """
        if self.model is None:
            self.init_model()

        # use provided prompt or default start
        start = prompt if prompt is not None else self.start

        # encode the beginning of the prompt
        if start.startswith("FILE:"):
            with open(start[5:], "r", encoding="utf-8") as f:
                start = f.read()
        start_ids = self.encode(start)
        x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]

        # run generation
        generated_texts = []
        with torch.no_grad():
            with self.ctx:
                for k in range(self.num_samples):
                    y = self.model.generate(
                        x, self.max_new_tokens, temperature=self.temperature, top_k=self.top_k
                    )
                    generated_text = self.decode(y[0].tolist())
                    generated_texts.append(generated_text)
                    print(generated_text)
                    print("---------------")

        return generated_texts
