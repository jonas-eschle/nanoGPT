"""
Utility functions for the GPT model.
"""

import os
import pickle
import numpy as np
import torch

def set_seed(seed):
    """
    Set random seed for reproducibility
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def setup_logging(out_dir):
    """
    Setup logging directory
    """
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def get_device(device_str=None):
    """
    Get the appropriate device
    """
    if device_str is None:
        # Auto-detect
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device_str

def get_dtype(dtype_str=None):
    """
    Get the appropriate dtype
    """
    if dtype_str is None:
        # Auto-detect
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return 'bfloat16'
        elif torch.cuda.is_available():
            return 'float16'
        else:
            return 'float32'
            
    return dtype_str

def load_meta(dataset_name):
    """
    Load metadata for a dataset
    """
    meta_path = os.path.join('data', dataset_name, 'meta.pkl')
    if not os.path.exists(meta_path):
        return None
        
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    return meta

def get_batch_iterator(data_path, batch_size, block_size, device):
    """
    Create a batch iterator for training or evaluation
    """
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    
    def get_batch():
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
        
    return get_batch