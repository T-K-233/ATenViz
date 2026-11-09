"""Llama models in bf16 (HuggingFace transformers), built on the meta device: no weights in memory.

    uv run model-viewer capture examples/llama.py:build -o out/tiny_llama
    uv run model-viewer capture examples/llama.py:build_7b -o out/llama7b
"""

import torch
from transformers import LlamaConfig, LlamaForCausalLM


def _build(config: LlamaConfig, tokens: int):
    with torch.device("meta"):
        model = LlamaForCausalLM(config).to(torch.bfloat16).eval()
    return model, (), {"input_ids": torch.zeros(1, tokens, dtype=torch.long, device="meta"), "use_cache": False}


def build():
    """Return a 2-layer Llama with a hidden size of 128, and a prompt of 16 tokens."""
    return _build(LlamaConfig(vocab_size=1000, hidden_size=128, intermediate_size=344, num_hidden_layers=2,
                              num_attention_heads=4, num_key_value_heads=2), 16)


def build_7b():
    """Return a Llama with the sizes of Llama-2-7B, and a prompt of 2048 tokens."""
    return _build(LlamaConfig(vocab_size=32000, hidden_size=4096, intermediate_size=11008, num_hidden_layers=32,
                              num_attention_heads=32, num_key_value_heads=32), 2048)
