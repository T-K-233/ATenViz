# Graph format (`atenviz.graph/1`)

`atenviz.capture` returns one JSON object, and `atenviz capture` writes it to a file. The
object is a dataflow graph in SSA form: each tensor has one producer op, or no producer for a graph
input or a weight. The ids are list indices.

```json
{
  "format": "atenviz.graph/1",
  "model": {"name": "LlamaForCausalLM", "torch_version": "2.15.0",
            "kept_ops": ["aten.scaled_dot_product_attention.default"]},
  "inputs": [1],
  "outputs": [227],
  "tensors": [...],
  "ops": [...],
  "modules": [...]
}
```

The ops are in execution order, which is the order of the `torch.export` graph. `inputs` and
`outputs` are the tensor ids of the user inputs and the user outputs. `kept_ops` are the ops that the
capture did not decompose to Core ATen.

## tensors

| Field | Meaning |
| --- | --- |
| `id`, `name` | The index, and the name: the parameter path for a weight (`model.layers.0.mlp.up_proj.weight`), the input name, or the FX node name. An op with more than one output names them `node.0`, `node.1`. |
| `shape`, `dtype`, `bytes` | The static shape, the short dtype name (`bf16`, `fp32`, `i64`, `fp8e4m3`, `u4`), and the bytes. The bytes of a dtype of 1 to 7 bits (`u4`, `i4`) are packed: `numel * bits / 8`, rounded up. torch stores one element in each byte. |
| `symbolic_shape` | Only for a tensor with a data-dependent size (from a boolean mask or `nonzero`): the shape with symbols, for example `["u0"]`. Then `shape` and `bytes` use the upper bound of each symbol. |
| `kind` | `input`, `param`, `buffer`, `constant` (a lifted tensor constant) or `activation` (an op output). |
| `const` | True for a weight, and for the output of a constant-foldable op. |
| `producer` | The id of the op that writes the tensor, or `null`. |
| `source` | For a constant: the parameters, buffers and constants that it comes from, separated by commas. |
| `storage` | For a constant: the id of the tensor whose memory it uses. A view of a weight (`permute`, `expand`) has the storage of the weight. Any other constant has its own storage. `null` for the other tensors. |

## ops

| Field | Meaning |
| --- | --- |
| `id`, `name` | The index and the FX node name. |
| `target` | The ATen overload, for example `aten.addmm.default`, or `atenviz.quantized_mm.default`. See [Quantized matmul](#quantized-matmul). |
| `module` | The id of the innermost module that runs the op. |
| `inputs`, `input_args` | The ids of the tensors that the op reads, and the schema argument name of each (`mat1`, `mat2`, `self`). A list argument (the tensors of `cat`) gives one entry for each tensor. |
| `outputs` | The ids of the tensors that the op writes. Tuple outputs are flat: there are no `getitem` ops. |
| `attrs` | The non-tensor arguments by schema name, for example `{"dim": [-1], "eps": 1e-05}`. Only the arguments of the call, not the defaults. A non-finite float is a string: `"inf"`, `"-inf"` or `"nan"`. |
| `flops` | `{"matmul": int, "nonmatmul": int}`. |
| `dtype` | The dtype pair, for example `bf16×bf16→bf16`. |
| `category` | `matmul` if `flops.matmul > 0`, else `nonmatmul` if `flops.nonmatmul > 0`, else `none`. |
| `view` | True if each output aliases an input (from the schema alias annotations). A view op moves no data. |
| `const` | True if each tensor input is a constant and the op is not random. A constant op is constant-foldable. |
| `weight_reads`, `weight_bytes` | The constant inputs of a non-constant op, and the bytes that it reads from them. 0 for a constant op. |
| `origin`, `origin_target` | The op before the decomposition, if the capture decomposed it (`linear`, `aten.linear.default`). |
| `source` | `file:line in function: code` of the innermost frame of the model code. The file is its import path: the path relative to the `sys.path` entry that holds it, for example `lerobot/policies/pi0/modeling_pi0.py`. Thus the graph has no local directories. |

### Quantized matmul

`atenviz.quantized_mm(input, weight, scales, biases, offs=None)` is a matmul with a quantized weight
(`atenviz.quantize`). The weight [N, K] has unsigned integers `q` of 1 to 8 bits (`u4`). `scales` and
`biases` are [N, K / G], one value for each group of G elements along K. The weight is
`scale * q + bias`, as in the affine mode of MLX, and the op returns `input @ weight.T`, as
`aten.linear`. With `offs`, the weight [E, N, K] has one matrix for each expert, and `offs` gives the
end row of each expert in `input`, as for `aten._grouped_mm`.

The dtype pair is `input×weight→output`, for example `bf16×u4→bf16`. The matmul FLOPs are
`2 * output elements * K`, and the dequantization is 2 non-matmul FLOPs for each weight element.

## modules

A module entry is one call of an `nn.Module`, or a box that groups other entries. When a model calls
a module two times, there are two entries, and the second has the name `shared@1`. The entries are
in depth-first order: a parent comes before its children. The children are in execution order.

| Field | Meaning |
| --- | --- |
| `id`, `parent`, `depth`, `children` | The tree. Id 0 is the root (the model). |
| `kind` | `module` for a module. `function` for the items that ran in one call of a Python function. `composite` for the ops that one composite op decomposed into (the `bmm`, `_softmax` and views of a decomposed attention). A composite has 2 or more ops that are not views or constants, and it is not a whole module. |
| `name`, `path`, `class` | For a module: the last part of the path (with `@N` for a repeated call), the attribute path from the root (`model.layers.0.self_attn`), and the qualified class name. For a function: the function name, `parent path/function`, and the function name. For a composite: the FX node name, `parent path/node`, and the composite op. |
| `call` | `false` for a module that owns module calls but that the model did not call (a `ModuleList`, or a layer whose parts a loop calls). There is no `call` field for the other entries. |
| `file` | For a function: its source file, as an import path (see the `source` of an op). |
| `ops` | The ids of the ops directly in the entry. |
| `inputs`, `outputs` | The activation tensors that cross the boundary of the entry: an op inside reads an input that an op outside writes (or that is a graph input), and an op outside reads an output that an op inside writes (or that is a graph output). Constant-foldable ops do not count. |
| `weights` | The constant tensors that the non-constant ops of the entry read. |
| `stats` | The sums over the non-constant ops in the entry and its descendants (see below). |

`stats`:

```json
{"flops": {"matmul": {"bf16×bf16→bf16": 5799936}, "nonmatmul": {"bf16→bf16": 18432, "fp32→fp32": 26368}},
 "weight_bytes": 367616, "weight_footprint": 365568, "ops": 102, "const_flops": 1024}
```

- `weight_bytes`: the sum of the `weight_bytes` of the ops.
- `weight_footprint`: the bytes of the unique storages of `weights`.
- `const_flops`: the FLOPs of the constant-foldable ops. The other numbers do not include them.
