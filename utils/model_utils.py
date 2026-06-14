"""
model_utils.py — Shared utilities for loading models and managing devices.

Cung cấp các hàm tiện ích dùng chung cho toàn bộ pipeline:
- Tải mô hình GPT-2 Small và tokenizer
- Tự động phát hiện thiết bị (CPU/GPU)
- Thiết lập random seed cho khả năng tái tạo kết quả
"""

import torch
import random
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Tuple


def set_seed(seed: int = 42) -> None:
    """
    Thiết lập random seed cho tất cả các thư viện để đảm bảo tái tạo kết quả.
    
    Args:
        seed: Giá trị seed (mặc định: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    Tự động phát hiện thiết bị tính toán tốt nhất.
    
    Returns:
        torch.device: cuda nếu có GPU, nếu không thì cpu
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_tokenizer(
    model_name: str = "gpt2",
    device: torch.device = None
) -> Tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    """
    Tải mô hình GPT-2 và tokenizer từ HuggingFace.
    
    GPT-2 Small (124M params, 12 layers, 12 heads, d_model=768) là mô hình
    benchmark chuẩn trong nghiên cứu Mechanistic Interpretability (Wang+22, 
    Conmy+23, Bricken+23).
    
    Args:
        model_name: Tên mô hình trên HuggingFace (mặc định: "gpt2")
        device: Thiết bị tính toán. Nếu None, tự động phát hiện.
        
    Returns:
        Tuple[GPT2LMHeadModel, GPT2Tokenizer]: (model, tokenizer)
    """
    if device is None:
        device = get_device()
    
    print(f"[*] Loading model '{model_name}' on {device}...")
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(model_name, attn_implementation="eager")
    model = model.to(device)
    model.eval()
    
    num_layers = model.config.n_layer
    num_heads = model.config.n_head
    d_model = model.config.n_embd
    
    print(f"[✓] Model loaded: {num_layers} layers, {num_heads} heads, d_model={d_model}")
    
    return model, tokenizer


def get_model_config(model: GPT2LMHeadModel) -> dict:
    """
    Trích xuất thông tin cấu hình quan trọng của mô hình.
    
    Args:
        model: Mô hình GPT-2 đã tải
        
    Returns:
        dict: Cấu hình gồm n_layer, n_head, n_embd, vocab_size
    """
    return {
        "n_layer": model.config.n_layer,
        "n_head": model.config.n_head,
        "n_embd": model.config.n_embd,
        "vocab_size": model.config.vocab_size,
    }
