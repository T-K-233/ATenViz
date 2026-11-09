# Survey: existing tools and what this tool reuses

We did this survey on 2026-11-09 with torch 2.15.0. "Checked" means that we did a local experiment
with this version. The other statements cite their source.

## 1. Capture the graph

| Approach | What it gives | Limits for this tool |
| --- | --- | --- |
| **`torch.export`** (used) | An ATen graph with the FakeTensor shapes and dtypes of each node (`meta["val"]`), the module call stack of each node (`meta["nn_module_stack"]`), the source line (`meta["stack_trace"]`), and a signature that marks the parameters, buffers and constants. `unflatten` makes the module hierarchy again from the same metadata ([unflatten source](https://docs.pytorch.org/docs/2.6/_modules/torch/export/unflatten.html)). | The model must be traceable. Control flow that depends on tensor values needs `torch.cond`. |
| `torch.fx.symbolic_trace` | A graph of module calls and functions. | No shapes or dtypes without a second pass, no ATen decomposition, and it fails on more Python code than `torch.export`. |
| `TorchDispatchMode` and module hooks (`FlopCounterMode`) | Each ATen op of an eager run, with the module from the forward hooks ([design note](https://dev-discuss.pytorch.org/t/the-ideal-pytorch-flop-counter-with-torch-dispatch/505), [docs](https://docs.pytorch.org/docs/2.15/generated/torch.utils.flop_counter.FlopCounterMode.html)). | Counts only. For a dataflow graph, you must track the tensor identity, the aliases and the in-place ops yourself. |
| torchview, torchinfo, fvcore | Module drawings (Graphviz, with a `depth` option), summaries, and FLOP counts from `jit.trace` ([torchview](https://pypi.org/project/torchview/), [fvcore](https://www.github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md)). | Static images or tables, with no ATen graph for a compiler. |
| ONNX (`torch.onnx.export(dynamo=True)`) | See section 3. | |

Checked with torch 2.15:

- The default output of `torch.export.export` is the training IR. It keeps `aten.linear`,
  `aten.layer_norm` and `aten.scaled_dot_product_attention`. This agrees with the export source
  ([v2.6 `_trace.py`](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/export/_trace.py#L1969)).
  `run_decompositions()` lowers the graph to Core ATen (the ops with the tag `core` in
  [native_functions.yaml](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml))
  and functionalizes the in-place ops. For example, `linear` becomes `permute` and `addmm`.
- When you delete an op from `torch.export.default_decompositions()`, the op stays whole.
- Each call of a module has its own key in `nn_module_stack`. The second call of `shared` has the key
  `L__self__shared@1`, so the calls stay separate.
- After the decomposition, `meta["from_node"]` names the op before the decomposition. The tool uses it
  to put the ops of a decomposed attention in one box.
- `torch.export` works on a model on the `meta` device. The capture of a bf16 Llama-2-7B (32 layers,
  2048 tokens) takes about 9 s with no weight memory.
- The Core ATen decomposition of a bf16 `scaled_dot_product_attention` is the math reference. It
  upcasts to fp32 and adds `bmm` ops in fp32. Thus the tool keeps attention whole by default.
- `torch.export` records only the stack frames of `forward` methods and of registered anchors
  (`_record_forward_stack_traces_only` and `_STACK_TRACE_ANCHORS` in `torch/fx/proxy.py`). Thus an
  op in `sample_actions` or `denoise_step` points at the nearest `forward`. During the export, the
  tool makes each frame of the model code an anchor.
- A `@torch.no_grad` method adds a `wrap_with_set_grad_enabled` subgraph to the training IR. The tool
  exports under `torch.no_grad()`, so the grad mode does not change and there is no subgraph.
- `run_decompositions()` inlines a `torch.autocast` region with the metadata of the region. The ops
  in the region keep their casts but lose their `nn_module_stack`.
- A boolean-mask index (in the SmolVLM vision embeddings) gives a data-dependent size: a symbol such
  as `u0` with a range of 0 to 1024. The tool counts it at its upper bound.

## 2. Count FLOPs

`torch.utils.flop_counter.flop_registry` maps ATen ops to FLOP formulas (mm, addmm, bmm,
convolution, the attention kernels). A formula takes tensors or FakeTensors, so the tool calls the
registry with the FakeTensors of the export. Checked: the matmul total of the tool is equal to the
total of `FlopCounterMode` on an eager run (unit test). The Llama-2-7B total (29.26 TFLOP) is equal
to 2 × the linear parameters × the tokens, plus the attention matmuls.

The registry has no formulas for elementwise ops, reductions or norms, so it counts them as 0. The
tool adds the rules in `model_viewer/costs.py`.

## 3. Is ONNX useful?

Checked with `torch.onnx.export(model, args, dynamo=True)` on a small transformer block:

- The exporter runs `torch.export` first and then translates ATen to ONNX
  ([docs](https://docs.pytorch.org/docs/2.15/onnx_export.html)). Each ONNX node has the FX node that
  it comes from (`pkg.torch.onnx.fx_node`) and the module hierarchy (`namespace`,
  `pkg.torch.onnx.class_hierarchy`), from `nn_module_stack`
  ([source](https://github.com/pytorch/pytorch/blob/main/torch/onnx/_internal/exporter/_core.py#L286-L348)).
  Thus ONNX has no information that the ExportedProgram does not have.
- The translation changes the ops: one `aten.linear` became `MatMul` and `Add`, and some nodes
  (`Split`) had no metadata.
- The export failed on a valid model: `scaled_dot_product_attention` with 3-D inputs gave "only 4D
  query, key, and value are supported".
- Tools exist: [onnx-tool](https://github.com/ThanatosShinji/onnx-tool/blob/main/data/Profile.md)
  shows the MACs and the memory of each node. [Netron](https://github.com/lutzroeder/netron/issues/1370)
  cannot collapse a graph by hierarchy.

Conclusion: ONNX is one translation step after the point where this tool captures the graph, and it
loses ATen detail. It does not help to capture or to count. It is useful as an exchange format if a
later step needs ONNX Runtime, or a tool that reads only ONNX. In that case, export ONNX next to this
graph. The same `nn_module_stack` paths connect the two.

## 4. Show the graph

| Tool | How it shows the hierarchy | Notes |
| --- | --- | --- |
| [Model Explorer](https://github.com/google-ai-edge/model-explorer) (Google AI Edge) | Layers from `/`-separated node namespaces. You click a layer to expand it. Each expanded layer has its own dagre layout in a Web Worker, and its size goes up to its parents ([graph_expander.ts](https://github.com/google-ai-edge/model-explorer/blob/b6726ac/src/ui/src/components/visualizer/worker/graph_expander.ts#L66)). WebGL rendering. | Its PyTorch adapter reads `nn_module_stack` from an ExportedProgram, but it does not decompose or count FLOPs ([adapter](https://github.com/google-ai-edge/model-explorer/blob/b6726ac/src/server/package/src/model_explorer/pytorch_exported_program_adapter_impl.py#L122-L249)). It opens a graph JSON file directly ([input_graph.ts](https://github.com/google-ai-edge/model-explorer/blob/b6726ac/src/ui/src/components/visualizer/common/input_graph.ts#L32-L230)), and `groupNodeAttributes` attaches values to a layer. Checked: Model Explorer opens the `--model-explorer` output of this tool. |
| Netron | A flat graph. | No hierarchical collapse (open request #1370). |
| torchview | Nested Graphviz clusters to a set depth. | A static image. |

Layout engines:

- [ELK Layered](https://eclipse.dev/elk/reference/algorithms/org-eclipse-elk-layered.html)
  (elkjs, EPL-2.0) supports layers from top to bottom, fixed port positions on a node
  (`portConstraints: FIXED_POS`), and constraints for the first and the last layer
  ([options](https://eclipse.dev/elk/reference/options/)). It is slow on large single graphs: about
  4 s for 406 nodes ([elkjs#372](https://github.com/kieler/elkjs/issues/372)). It can also overflow
  the stack ([elkjs#314](https://github.com/kieler/elkjs/issues/314)), and it has no collapse or
  expand ([elkjs#336](https://github.com/kieler/elkjs/issues/336)).
- [dagre](https://github.com/dagrejs/dagre) (MIT, maintained as `@dagrejs/dagre`) is smaller, but it
  has no port constraints.
- Checked on SmolVLA: ELK overflowed the stack on a module with 167 items and 518 edges. On the 84
  items of the action-expert layers, network-simplex node placement took 21.6 s and Brandes-Koepf
  placement took 0.24 s. Thus the viewer uses network-simplex only for modules of 40 items or fewer.
  It also routes one ELK edge for each pair of connected items, and it tries simpler settings when a
  layout fails.

## 5. Models

- SmolVLA, pi0 and pi0.5 come from LeRobot 0.6.1. One generic builder
  (`examples/lerobot_policy.py`) captures all three from their checkpoint configs.
- pi0.7 has no public code or weights. Physical Intelligence announced it on 2026-04-16 with a
  [paper](https://www.pi.website/download/pi07.pdf). [openpi](https://github.com/Physical-Intelligence/openpi)
  has pi0, pi0-FAST and pi0.5 only, and LeRobot has no pi0.7 policy.

## What this tool reuses

- From PyTorch: `torch.export`, `run_decompositions` with a decomposition table that you can change,
  the graph signature, `nn_module_stack`, `from_node`, `stack_trace`, the FLOP formulas of
  `torch.utils.flop_counter`, and the schema alias annotations (which ops are views).
- From Model Explorer: layers that open in place, and a layout for each layer whose size goes up to
  its parents. The tool also writes the graph JSON of Model Explorer (`--model-explorer`), with the
  module totals as layer attributes.
- ELK Layered for the layout of each open module (elkjs 0.11.0, in the package). The viewer caches
  the layouts by structure, so the 32 equal layers of Llama-2-7B get one layout. Each layout is small,
  and thus the elkjs limits above do not apply.
