"""Quantized weights, as a deployment stores them.

A quantized weight has ``bits``-bit unsigned integers ``q``. Each group of ``group_size`` elements
along the input dimension has one scale and one bias, and the weight is ``scale * q + bias``. This is
the affine mode of MLX.

ATenViz captures only shapes and dtypes, so the quantized modules have empty weights. torch has no
kernels for the dtypes of 1 to 7 bits, so a model with 4-bit weights runs only on the meta device.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@torch.library.custom_op("atenviz::quantized_mm", mutates_args=())
def quantized_mm(input: Tensor, weight: Tensor, scales: Tensor, biases: Tensor, offs: Tensor | None = None) -> Tensor:
    """Multiply ``input`` [..., K] by the transposed weight [N, K], as ``F.linear``.

    With ``offs``, the weight [E, N, K] has one matrix for each expert of a mixture-of-experts layer,
    and ``offs`` gives the end row of each expert in ``input``, as for ``torch._grouped_mm``.
    """
    w = weight.to(scales.dtype).unflatten(-1, (scales.shape[-1], -1))
    w = (w * scales[..., None] + biases[..., None]).flatten(-2)
    return F.linear(input, w) if offs is None else torch._grouped_mm(input, w.transpose(-2, -1), offs=offs)


@quantized_mm.register_fake
def _(input, weight, scales, biases, offs=None):
    return input.new_empty(*input.shape[:-1], weight.shape[-2])


def _add_weights(module: nn.Module, weight: Tensor, bits: int, group_size: int) -> None:
    *rows, k = weight.shape
    if k % group_size:
        raise ValueError(f"atenviz: the input dimension {k} is not a multiple of the group size {group_size}")

    def empty(shape, dtype):
        return nn.Parameter(torch.empty(shape, dtype=dtype, device=weight.device), requires_grad=False)

    module.weight = empty(weight.shape, getattr(torch, f"uint{bits}"))
    module.scales = empty((*rows, k // group_size), weight.dtype)
    module.biases = empty((*rows, k // group_size), weight.dtype)


class QuantizedLinear(nn.Module):
    """``nn.Linear`` with a quantized ``weight``. A 3-D weight has one matrix for each expert: then
    give ``offs`` to ``forward``."""

    def __init__(self, weight: Tensor, bits: int, group_size: int, bias: Tensor | None = None):
        super().__init__()
        _add_weights(self, weight, bits, group_size)
        self.bias = bias

    def forward(self, x: Tensor, offs: Tensor | None = None) -> Tensor:
        y = quantized_mm(x, self.weight, self.scales, self.biases, offs)
        return y if self.bias is None else y + self.bias


class QuantizedEmbedding(nn.Module):
    """``nn.Embedding`` with a quantized ``weight``. A lookup reads and dequantizes only its rows."""

    def __init__(self, weight: Tensor, bits: int, group_size: int):
        super().__init__()
        _add_weights(self, weight, bits, group_size)

    def forward(self, ids: Tensor) -> Tensor:
        rows = F.embedding(ids, self.weight).to(self.scales.dtype).unflatten(-1, (self.scales.shape[-1], -1))
        return (rows * F.embedding(ids, self.scales)[..., None] + F.embedding(ids, self.biases)[..., None]).flatten(-2)


def quantize(model: nn.Module, bits: int = 4, group_size: int = 64) -> nn.Module:
    """Replace each ``nn.Linear`` and ``nn.Embedding`` under ``model`` with its quantized form, in place."""
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            setattr(model, name, QuantizedLinear(child.weight, bits, group_size, child.bias))
        elif isinstance(child, nn.Embedding):
            setattr(model, name, QuantizedEmbedding(child.weight, bits, group_size))
        else:
            quantize(child, bits, group_size)
    return model
