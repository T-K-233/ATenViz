"""Capture a PyTorch model as a hierarchical ATen graph, and write the graph and its viewer page."""

import argparse
import ast
import importlib
import importlib.util
import json
import math
import sys
from pathlib import Path

from .capture import DEFAULT_KEEP_OPS, capture
from .model_explorer import to_model_explorer
from .render import render_html
from .traffic import by_module, load, traffic


def parse_size(text: str) -> float:
    """``0``, ``inf``, a byte count, or a size with a binary unit: ``512K``, ``32M``, ``1G``."""
    if text == "inf":
        return math.inf
    scale = {"K": 2**10, "M": 2**20, "G": 2**30}.get(text[-1:].upper(), 1)
    return float(text[:-1] if scale > 1 else text) * scale


def load_function(spec: str):
    """Return the function that ``spec`` names: ``FILE.py:FUNCTION`` or ``MODULE:FUNCTION``."""
    target, _, name = spec.rpartition(":")
    if not target or not name:
        raise SystemExit(f"Expected FILE.py:FUNCTION or MODULE:FUNCTION, got {spec!r}")
    if not target.endswith(".py"):
        return getattr(importlib.import_module(target), name)
    path = Path(target).resolve()
    sys.path.insert(0, str(path.parent))
    module_spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return getattr(module, name)


def parse_arg(text: str) -> tuple[str, object]:
    """Parse ``KEY=VALUE``. The value is a Python literal if it parses as one, else a string."""
    key, sep, value = text.partition("=")
    if not sep:
        raise SystemExit(f"Expected KEY=VALUE, got {text!r}")
    try:
        return key, ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return key, value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="atenviz", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capture", help="Export a model and write OUT.json (the graph) and OUT.html (the viewer).")
    cap.add_argument("spec", help="FILE.py:FUNCTION or MODULE:FUNCTION. The function returns (model, args) or "
                                  "(model, args, kwargs), with example inputs.")
    cap.add_argument("-a", "--arg", action="append", default=[], metavar="KEY=VALUE",
                     help="Keyword argument for the function, for example checkpoint=lerobot/pi0_base. "
                          "Repeat for more.")
    cap.add_argument("-o", "--out", type=Path, help="Output path without extension. Default: the model name.")
    cap.add_argument("--name", help="Model name for the viewer. Default: the model class name.")
    cap.add_argument("--keep-op", action="append", metavar="OP",
                     help=f"Do not decompose OP. Repeat for more ops. Default: {', '.join(DEFAULT_KEEP_OPS)}.")
    cap.add_argument("--decompose-all", action="store_true", help="Decompose all ops to Core ATen.")
    cap.add_argument("--model-explorer", action="store_true", help="Also write OUT.model_explorer.json.")
    ren = sub.add_parser("render", help="Write the viewer page for graph files. With more than one graph, the "
                                         "page has a menu.")
    ren.add_argument("graph", type=Path, nargs="+")
    ren.add_argument("-o", "--out", type=Path, help="Output page. Default: the first graph path with .html.")
    ren.add_argument("--label", action="append", help="Menu item of each graph, in graph order. Default: the "
                                                      "model name without its parenthesized part.")
    ren.add_argument("--group", action="append", help="Menu group of each graph, in graph order, for example "
                                                      "HALO. Default: no groups.")
    ren.add_argument("--model-explorer", action="store_true",
                     help="Also write OUT.model_explorer.json for the first graph.")
    tra = sub.add_parser("traffic", help="Print the DRAM traffic and the operational intensity of a graph for "
                         "each on-chip capacity (atenviz/traffic.py has the rules).")
    tra.add_argument("graph", type=Path, help="A graph: .json or .json.gz.")
    tra.add_argument("-c", "--capacity", action="append", type=parse_size, metavar="SIZE",
                     help="On-chip memory: 0, 512K, 32M, inf. Repeat for more. Default: 0, 32M and inf.")
    tra.add_argument("--fusion", choices=("none", "vector", "epilogue"), default="epilogue")
    tra.add_argument("--writeback", choices=("live", "all"), default="live")
    tra.add_argument("--act-dtype", help="The dtype of the activations, inputs and outputs, for example bf16.")
    tra.add_argument("--weight-dtype", action="append", metavar="[GLOB=]DTYPE",
                     help="The dtype of the weights, or of the weights whose name matches GLOB. A number is bits "
                          "per element (4.5). Repeat for more; the first match wins.")
    tra.add_argument("--by-module", type=int, metavar="DEPTH", help="Also sum the kernels by module at DEPTH.")
    args = parser.parse_args(argv)
    if args.cmd == "traffic":
        return print_traffic(args)
    for option in ("label", "group"):
        if args.cmd == "render" and getattr(args, option) and len(getattr(args, option)) != len(args.graph):
            parser.error(f"give one --{option} for each graph")

    if args.cmd == "capture":
        model, model_args, *model_kwargs = load_function(args.spec)(**dict(map(parse_arg, args.arg)))
        keep = () if args.decompose_all else tuple(args.keep_op or DEFAULT_KEEP_OPS)
        graphs = [capture(model, model_args, model_kwargs[0] if model_kwargs else None, keep, args.name)]
        out = args.out or Path(graphs[0]["model"]["name"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_name(out.name + ".json").write_text(json.dumps(graphs[0]))
        print(f"wrote {out}.json  ({len(graphs[0]['ops'])} ops, {len(graphs[0]['modules'])} modules)")
        slugs = labels = groups = None
    else:
        graphs = [json.loads(p.read_text()) for p in args.graph]
        out = (args.out or args.graph[0]).with_suffix("")
        slugs, labels, groups = [p.stem for p in args.graph], args.label, args.group
    out.with_name(out.name + ".html").write_text(render_html(graphs, slugs, labels, groups))
    print(f"wrote {out}.html")
    if args.model_explorer:
        out.with_name(out.name + ".model_explorer.json").write_text(json.dumps(to_model_explorer(graphs[0])))
        print(f"wrote {out}.model_explorer.json")


def print_traffic(args) -> None:
    def dtype(text: str):
        try:
            return float(text)
        except ValueError:
            return text

    weight = None
    for item in args.weight_dtype or ():
        glob, _, d = item.rpartition("=")
        weight = weight or {}
        weight[glob or "*"] = dtype(d)
    graph = load(args.graph)
    for cap in args.capacity or (0, 32 * 2**20, math.inf):
        r = traffic(graph, cap, args.fusion, args.writeback, args.act_dtype, weight)
        print(f"capacity {cap:g} B: {r['flops'] / 1e9:.3f} GFLOP, reads {r['reads'] / 1e9:.4f} GB "
              f"(weights {r['weight_reads'] / 1e9:.4f}), writes {r['writes'] / 1e9:.4f} GB, "
              f"{r['intensity']:.2f} FLOP/B, {len(r['kernels'])} kernels")
        if args.by_module is not None:
            for name, s in sorted(by_module(graph, r, args.by_module).items(), key=lambda x: -x[1]["reads"] - x[1]["writes"]):
                print(f"  {name:40s} reads {s['reads'] / 1e9:9.4f} GB  writes {s['writes'] / 1e9:9.4f} GB  "
                      f"{s['flops'] / 1e9:10.3f} GFLOP  {s['kernels']} kernels")
