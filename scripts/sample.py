#!/usr/bin/env python
"""
Script for sampling from a trained GPT model
"""

import os
import sys

# Add the parent directory to the path so we can import nanogpt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nanogpt.config import Config
from nanogpt.sampler import Sampler


def main():
    # Create default configuration
    config = Config(
        init_from="resume",
        out_dir="out",
        start="\n",
        num_samples=10,
        max_new_tokens=500,
        temperature=0.8,
        top_k=200,
        seed=1337,
        device="cuda",
        dtype="bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16",
        compile=False,
    )

    # Override with command line arguments or config file
    config = Config.from_command_line(config)

    # Create sampler and generate text
    sampler = Sampler(config.to_dict())
    sampler.generate()


if __name__ == "__main__":
    import torch

    main()
