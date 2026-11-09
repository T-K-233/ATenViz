"""FLOP counts of ATen ops.

One FLOP is one scalar arithmetic operation. A multiply-accumulate is 2 FLOPs, and a transcendental
function (exp, erf, rsqrt) is 1 FLOP. Each op has two counts:

- matmul: the contraction of a matmul, convolution or attention op (2 * M * N * K). The matmul and
  convolution formulas come from ``torch.utils.flop_counter``.
- nonmatmul: all other arithmetic, which includes the bias add of addmm and the softmax of attention.

A view op or a data-movement op (copy, cat, gather, dtype cast) is 0 FLOPs. An op that is not in a
table below is 1 FLOP for each output element.
"""

import math

import torch
from torch.utils._pytree import tree_flatten
from torch.utils.flop_counter import flop_registry

aten = torch.ops.aten

DTYPE_NAMES = {
    torch.float64: "fp64", torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16",
    torch.float8_e4m3fn: "fp8e4m3", torch.float8_e5m2: "fp8e5m2", torch.int64: "i64", torch.int32: "i32",
    torch.int16: "i16", torch.int8: "i8", torch.uint8: "u8", torch.bool: "bool", torch.complex64: "c64",
    torch.complex128: "c128",
}

# The two operands of each matmul op, by schema argument name. Attention ops use (query, key).
MATMUL_OPERANDS = {
    aten.mm: ("self", "mat2"), aten.bmm: ("self", "mat2"), aten.addmm: ("mat1", "mat2"),
    aten.baddbmm: ("batch1", "batch2"), aten._scaled_mm: ("self", "mat2"), aten.linear: ("input", "weight"),
    aten.convolution: ("input", "weight"), aten._convolution: ("input", "weight"),
}

# Ops that create, copy, cast, gather or reorder data.
DATA_MOVEMENT = {
    aten.clone, aten._to_copy, aten.copy, aten.copy_, aten.cat, aten.stack, aten.constant_pad_nd, aten.index,
    aten.index_select, aten.gather, aten.embedding, aten.full, aten.full_like, aten.zeros, aten.zeros_like,
    aten.ones, aten.ones_like, aten.empty, aten.empty_like, aten.empty_strided, aten.arange, aten.scalar_tensor,
    aten.lift_fresh_copy, aten.repeat, aten.flip, aten.roll, aten.scatter, aten.index_put, aten.slice_scatter,
    aten.select_scatter, aten.diagonal_scatter, aten.as_strided_scatter, aten.pixel_shuffle,
    aten.pixel_unshuffle, aten._unsafe_view, aten.tril, aten.triu, aten.resize_, aten.fill, aten.fill_,
}

# Ops that read only the elements they select. A gather from a weight (an embedding lookup) reads as
# many bytes as it writes.
GATHER = {aten.embedding, aten.index_select, aten.index, aten.gather, aten.take}

# Ops that read each input element once: 1 FLOP for each input element.
REDUCTIONS = {
    aten.sum, aten.mean, aten.amax, aten.amin, aten.max, aten.min, aten.prod, aten.any, aten.all, aten.argmax,
    aten.argmin, aten.cumsum, aten.cumprod, aten.logsumexp, aten.norm, aten.linalg_vector_norm,
    aten._adaptive_avg_pool2d, aten.adaptive_avg_pool2d,
}

# Ops that make several passes over their input: FLOPs = passes * input elements.
PASSES = {
    aten._softmax: 5, aten._log_softmax: 5, aten._safe_softmax: 5,  # max, subtract, exp, sum, divide
    aten.var: 3, aten.var_mean: 4, aten.std: 3,  # mean, subtract and square, sum
}


def _numel(t: torch.Tensor) -> int:
    return math.prod(t.shape)


def _affine(args: dict) -> int:
    return (args.get("weight") is not None) + (args.get("bias") is not None)


# Non-matmul ops with their own formula: (arguments by name, output) -> FLOPs. A norm makes 5 passes
# (mean, subtract, square, sum, scale) and 1 pass for each affine term.
FORMULAS = {
    aten.native_layer_norm: lambda a, out: _numel(a["input"]) * (5 + _affine(a)),
    aten.layer_norm: lambda a, out: _numel(a["input"]) * (5 + _affine(a)),
    aten.native_group_norm: lambda a, out: _numel(a["input"]) * (5 + _affine(a)),
    aten.rms_norm: lambda a, out: _numel(a["input"]) * (4 + _affine(a)),
    aten._native_batch_norm_legit_no_training: lambda a, out: _numel(a["input"]) * (2 + _affine(a)),
    aten.max_pool2d_with_indices: lambda a, out: _numel(out[0]) * math.prod(a["kernel_size"]),
    aten.avg_pool2d: lambda a, out: _numel(out) * math.prod(a["kernel_size"]),
}


def dtype_name(dtype: torch.dtype) -> str:
    return DTYPE_NAMES.get(dtype, str(dtype))


def is_view(op: torch._ops.OpOverload) -> bool:
    """Return True if every output of ``op`` aliases an input, so that the op moves no data."""
    returns = op._schema.returns
    return bool(returns) and all(r.alias_info is not None and not r.alias_info.is_write for r in returns)


def op_flops(op: torch._ops.OpOverload, args: dict, positional: list, kwargs: dict, out) -> tuple[int, int]:
    """Return the (matmul, nonmatmul) FLOPs of one op.

    ``args`` maps the schema argument names to the values, ``positional`` and ``kwargs`` are the call
    arguments, and ``out`` is the output. The tensors are FakeTensors or real tensors.
    """
    packet = op.overloadpacket
    if is_view(op) or packet in DATA_MOVEMENT:
        return 0, 0
    if "query" in args:  # scaled_dot_product_attention and its kernels
        *batch, length, embed = args["query"].shape
        source, value = args["key"].shape[-2], args["value"].shape[-1]
        scores = math.prod(batch) * length * source
        masked = args.get("attn_mask") is not None or bool(args.get("is_causal"))
        return 2 * scores * (embed + value), scores * (6 + masked)  # scale, softmax, mask
    outputs = [t for t in tree_flatten(out)[0] if isinstance(t, torch.Tensor)]
    if packet is aten.linear:
        return 2 * _numel(out) * args["weight"].shape[-1], _numel(out) if args.get("bias") is not None else 0
    if packet in flop_registry:
        biased = packet in (aten.addmm, aten.baddbmm) or args.get("bias") is not None
        return int(flop_registry[packet](*positional, **kwargs, out_val=out)), _numel(outputs[0]) if biased else 0
    if packet in FORMULAS:
        return 0, int(FORMULAS[packet](args, out))
    first = next((v for v in positional if isinstance(v, torch.Tensor)), None)
    if packet in PASSES and first is not None:
        return 0, PASSES[packet] * _numel(first)
    if packet in REDUCTIONS and first is not None:
        return 0, _numel(first)
    return 0, sum(_numel(t) for t in outputs)


def dtype_pair(packet, arg_names: list[str], inputs: list[str], outputs: list[str]) -> str:
    """Return the dtype pair of an op, for example ``bf16×bf16→fp32`` or ``bf16,fp32→fp32``.

    ``arg_names`` and ``inputs`` give the schema argument name and the dtype name of each tensor input.
    """
    operands = ("query", "key") if "query" in arg_names else MATMUL_OPERANDS.get(packet, ())
    if operands and all(n in arg_names for n in operands):
        return "×".join(inputs[arg_names.index(n)] for n in operands) + "→" + outputs[0]
    return ",".join(dict.fromkeys(inputs)) + "→" + outputs[0]
