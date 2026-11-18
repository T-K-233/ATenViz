"""Capture a PyTorch model as a hierarchical ATen graph and view it in the browser."""

import importlib.util

# Without torch, only atenviz.traffic works: it reads a graph file in plain Python.
if importlib.util.find_spec("torch") is not None:
    from .capture import build_graph, capture, capture_program
    from .model_explorer import to_model_explorer
    from .quant import QuantizedEmbedding, QuantizedLinear, quantize
    from .render import render_html

__all__ = ["QuantizedEmbedding", "QuantizedLinear", "build_graph", "capture", "capture_program", "quantize",
           "render_html", "to_model_explorer"]
