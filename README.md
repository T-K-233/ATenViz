# ATenViz

ATenViz (A Tensor graph Visualizer) captures a PyTorch model as a hierarchical ATen graph, counts
its FLOPs and its memory traffic, and shows the graph in the browser.

**[Open the viewer](https://t-k-233.github.io/ATenViz/)** with the graphs of π0, π0.5, SmolVLA,
[HALO](https://github.com/apple-aiml-research/ml-halo) and
[GR00T 2](https://research.nvidia.com/labs/gear/gr00t-2/).

## Install

ATenViz needs Python 3.12 or later and PyTorch 2.15 or later.

```bash
pip install .            # the package and the atenviz command
uv sync --group dev      # for development: also pytest, torchvision and transformers
```

## Use

Write a function that returns a model and its example inputs: `(model, args)` or
`(model, args, kwargs)`. A model on the `meta` device works.

```bash
atenviz capture examples/llama.py:build -o out/tiny_llama  # writes out/tiny_llama.json and .html
atenviz render out/a.json out/b.json -o out/page.html       # one page with a menu of graphs
atenviz traffic out/tiny_llama.json                         # DRAM traffic and operational intensity
```

```python
from pathlib import Path

from atenviz import capture, render_html

graph = capture(model, (x,))                      # a JSON-ready dict
Path("model.html").write_text(render_html([graph]))
```

## Examples

| Function | Model |
| --- | --- |
| `examples/llama.py:build`, `build_7b` | A 2-layer Llama in bf16, and a Llama with the sizes of Llama-2-7B. |
| `examples/resnet.py:build` | torchvision ResNet-18. |
| `examples/lerobot_policy.py:build` | A LeRobot policy: SmolVLA, π0 or π0.5. |
| `examples/halo.py:build_generator`, `build_tracker`, `build_orchestrator` | The three models of HALO. |
| `examples/gr00t2.py:build`, `build_sonic` | GR00T 2 (Turbo on the G1, and YAM) and SONIC. |

Each file has its commands and dependencies. `examples/graphs/` has the graphs of the viewer site.

## Docs

- [docs/guide.md](docs/guide.md): the capture options, the viewer, the counting rules for FLOPs and
  traffic, quantized weights, the example graphs and the limits.
- [docs/ir.md](docs/ir.md): the graph format.
- [docs/survey.md](docs/survey.md): the existing tools, and why the tool uses `torch.export` and not
  ONNX.

## Test

```bash
uv run pytest
```

## License

ATenViz is under the MIT license. See [LICENSE](LICENSE). The package also contains
`atenviz/viewer/vendor/elk.bundled.js`, which is elkjs 0.11.0 under the EPL-2.0. See
[atenviz/viewer/vendor/README.md](atenviz/viewer/vendor/README.md).
