# ATenViz

ATenViz (A Tensor graph Visualizer) captures a PyTorch model as a hierarchical ATen graph, counts
its FLOPs and its weight traffic, and shows the graph in the browser.

**[Open the viewer](https://t-k-233.github.io/ATenViz/)** with the graphs of SmolVLA, π0 and π0.5.

- **The graph.** `atenviz.capture` runs `torch.export` once and lowers the result to Core ATen.
  The export traces with FakeTensors, so a model on the `meta` device works and no op runs. The
  graph is one JSON file with the tensors, the ops and the tree of module calls. A compiler can read
  it: see [docs/ir.md](docs/ir.md).
- **The viewer.** The viewer is one HTML page that works offline. It shows the modules as nested
  boxes. The data flows from top to bottom, and independent ops are side by side. Zoom in on a
  module to open it, and zoom out to close it. Each op shows its output shape, its dtype, its FLOPs
  and the weight bytes that it reads. Each module shows its inputs, its outputs, its matmul and
  non-matmul FLOPs for each dtype pair, and the weight bytes that its ops read.

For the survey of the existing tools, and for the reasons to use `torch.export` and not ONNX, see
[docs/survey.md](docs/survey.md).

## Install

ATenViz needs Python 3.12 or later and PyTorch 2.15 or later.

```bash
pip install .            # the package and the atenviz command
uv sync --group dev      # for development: also pytest, torchvision and transformers
```

## Capture a model

To capture a model:

1. Write a function that returns the model and its example inputs: `(model, args)` or `(model, args, kwargs)`.
2. Run `atenviz capture FILE.py:FUNCTION -o OUT`.
3. Open `OUT.html` in a browser.

```bash
atenviz capture examples/llama.py:build -o out/tiny_llama
# wrote out/tiny_llama.json  (228 ops, 43 modules)
# wrote out/tiny_llama.html
```

`FUNCTION` can also be `MODULE:FUNCTION`. The examples are:

| Function | Model |
| --- | --- |
| `examples/llama.py:build` | A 2-layer Llama in bf16. |
| `examples/llama.py:build_7b` | A Llama with the sizes of Llama-2-7B. The capture takes about 9 s. |
| `examples/resnet.py:build` | torchvision ResNet-18. |
| `examples/lerobot_policy.py:build` | A LeRobot policy, for example SmolVLA, pi0 or pi0.5. See the file for the LeRobot install. |

The LeRobot policies capture one chunk of 50 actions with 10 flow-matching steps. This is 24,000 to
29,000 ops, and each capture takes 70 to 90 s:

```bash
atenviz capture examples/lerobot_policy.py:build -a checkpoint=lerobot/smolvla_base \
    -o out/smolvla --name "SmolVLA (lerobot/smolvla_base)"
atenviz capture examples/lerobot_policy.py:build -a checkpoint=lerobot/pi0_base -a dtype=bfloat16 \
    -o out/pi0 --name "π0 (lerobot/pi0_base, bf16)"
atenviz capture examples/lerobot_policy.py:build -a checkpoint=lerobot/pi05_base -a dtype=bfloat16 \
    -o out/pi05 --name "π0.5 (lerobot/pi05_base, bf16)"
atenviz render out/smolvla.json out/pi0.json out/pi05.json -o out/vla.html
```

| Option | Effect |
| --- | --- |
| `-a KEY=VALUE` | Give a keyword argument to the function. The value is a Python literal or a string. Repeat for more arguments. |
| `-o OUT` | Write `OUT.json` and `OUT.html`. The default is the model name. |
| `--name NAME` | Show `NAME` as the model name. The default is the class name of the model. |
| `--keep-op OP` | Do not decompose `OP`. Repeat for more ops. The default is `aten.scaled_dot_product_attention.default`. |
| `--decompose-all` | Decompose all ops to Core ATen, attention too. |
| `--model-explorer` | Also write `OUT.model_explorer.json` for [Model Explorer](https://github.com/google-ai-edge/model-explorer). |

Attention stays one op by default. Its Core ATen decomposition is the math reference, which upcasts
bf16 and fp16 attention to fp32. As a result, the dtypes of a decomposed attention are wrong for
hardware.

`atenviz render GRAPH.json [GRAPH.json ...] -o PAGE.html` writes the viewer page for graph
files. With more than one graph, the page has a picker, and the page address selects a graph
(`PAGE.html#pi0`). The viewer also opens a graph file (`.json` or `.json.gz`) from its Open file
button, or from a file that you drop on it.

From Python:

```python
from pathlib import Path

from atenviz import capture, render_html

graph = capture(model, (x,))                      # a JSON-ready dict
Path("model.html").write_text(render_html([graph]))
```

`capture_program` returns the `ExportedProgram`. `build_graph` converts an `ExportedProgram` that you
export yourself, and `to_model_explorer` converts a graph for Model Explorer.

## Example graphs

`examples/graphs/` has the graphs of SmolVLA, π0 and π0.5 from the LeRobot commands above (LeRobot
0.6.1, torch 2.15). Each file is gzip-compressed JSON. The viewer opens the file directly.

The workflow `.github/workflows/pages.yml` deploys the viewer to GitHub Pages on each push to `main`.
The page has a picker for the example graphs. It loads each graph from raw.githubusercontent.com at
the commit of the deployment. To enable the site, set Settings > Pages > Source to GitHub Actions.

To add or update an example graph:

1. Capture the model with `-o out/SLUG`.
2. Run `gzip -9n -c out/SLUG.json > examples/graphs/SLUG.json.gz`. With `-n`, the same graph gives the same file.
3. For a new graph, add `SLUG:LABEL` to `EXAMPLES` in `.github/workflows/pages.yml`.

Do not store the graphs with Git LFS. raw.githubusercontent.com sends the LFS pointer file, not the
graph.

## Use the viewer

| Action | Result |
| --- | --- |
| Scroll or pinch | Zoom. When you zoom in on a closed module (a card with `+`), it opens in place. When you zoom out of a small open module, it closes. |
| Drag | Pan. |
| Click | Select a module, an op, a module input or output, or a wire. The details panel shows the selection. The wires of the selected op are red. |
| Double-click | Open or close a module, or zoom to an op. |
| Tree (left) | Click a row to go to its module. Click the arrow of a row to open or close the module. The bar shows the share of the model FLOPs: matmul (blue) and non-matmul (orange). |
| Views, Constants | Show the view ops and the constant-foldable ops. By default, the viewer hides them. |
| `/`, `F`, `Esc` | Search, fit the graph, clear the selection. In the search field, press Enter to go to the next match. |

The page uses the FangXing II design system: black and white, one red for the selection, and the
chart colors for matmul (blue) and non-matmul (orange).

## How the tool counts

- **FLOPs.** A multiply-accumulate is 2 FLOPs. The matmul and convolution formulas come from
  `torch.utils.flop_counter`, so the totals agree with `FlopCounterMode`. The bias add of addmm and
  the softmax of attention are non-matmul FLOPs. An elementwise op is 1 FLOP for each output
  element, and a reduction is 1 FLOP for each input element. Softmax is 5 FLOPs for each element, and
  a layer norm is 5 to 7. View ops and data-movement ops (copy, cat, gather, dtype cast) are 0 FLOPs.
  `atenviz/costs.py` has the tables.
- **Dtype pair.** For a matmul op, the dtype pair is the dtypes of its two operands and of its output
  (`bf16×bf16→fp32`). For other ops, it is the dtypes of its tensor inputs and of its output
  (`bf16,fp32→fp32`).
- **Weights.** The weights are the parameters, the buffers and the lifted constants. An op that reads
  only weights is constant-foldable, for example a transposed weight, a rotary table that an op makes
  from a buffer, or a causal mask. The output of a constant-foldable op counts as a weight, and its
  FLOPs are not in the totals. "Weights read" is the sum of the weight bytes that each op reads. A
  view of a weight reads at most the bytes of the weight, and an embedding lookup reads only the rows
  that it selects. "Unique" counts each weight once.

## Module hierarchy

The boxes are module calls from the `nn_module_stack` of `torch.export`. When a model calls a module
two times, the viewer shows two boxes (`shared` and `shared@1`).

A model can call the parts of a module from its own code. For example, SmolVLA runs
`layers[i].self_attn.q_proj` and `layers[i].mlp` from one loop, and nothing calls a `ModuleList`.
Then the tool adds the modules that own the calls (`layers`, `layers.3`, `self_attn`), with their
classes, from the attribute paths. Consecutive calls under one owner share one box. An op between two
calls goes into the deepest box that holds both calls.

A model also does much of its work in plain methods, for example `embed_prefix`, each `denoise_step`
of a flow-matching loop, or `apply_rotary_pos_emb`. In each module, consecutive items that ran in one
call of a function go into a function box with a dotted border. The tool finds the functions from the
stack trace of each op. `torch.export` records only the frames of `forward` methods, so the capture
makes it record all frames of the model code. As a result, the `source` of each op is its own line of
model code.

## Limits

- `torch.export` must trace the model. The non-strict export of the tool handles most Python code.
  Control flow that depends on tensor values needs `torch.cond` or a change to the model. The shapes
  are static: the graph is for the example inputs.
- A data-dependent size (from a boolean mask or `nonzero`) has no value at export time. The tool counts
  it at the upper bound of its range and keeps the symbol in `symbolic_shape`.
- The export runs under `torch.no_grad()`. The ops in a `torch.autocast` region keep their casts, but
  they lose their module box, because `run_decompositions` gives them the metadata of the region.
- The non-matmul FLOPs follow the rules above. They are not a cost model of a specific hardware.

## Test

```bash
uv run pytest
```

## License

ATenViz is under the MIT license. See [LICENSE](LICENSE). The package also contains
`atenviz/viewer/vendor/elk.bundled.js`, which is elkjs 0.11.0 under the EPL-2.0. See
[atenviz/viewer/vendor/README.md](atenviz/viewer/vendor/README.md).
