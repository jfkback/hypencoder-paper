import torch


def dtype_lookup(dtype: str):
    dtype_lookup = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }
    return dtype_lookup[dtype]
