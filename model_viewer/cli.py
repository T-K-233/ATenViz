"""Capture a PyTorch model as a hierarchical ATen graph, and write the graph and its viewer page."""

import argparse
import ast
import importlib
import importlib.util
import json
import sys
from pathlib import Path

from .capture import DEFAULT_KEEP_OPS, capture
from .model_explorer import to_model_explorer
from .render import render_html


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
    parser = argparse.ArgumentParser(prog="model-viewer", description=__doc__)
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
                                         "page has a picker.")
    ren.add_argument("graph", type=Path, nargs="+")
    ren.add_argument("-o", "--out", type=Path, help="Output page. Default: the first graph path with .html.")
    ren.add_argument("--model-explorer", action="store_true",
                     help="Also write OUT.model_explorer.json for the first graph.")
    args = parser.parse_args(argv)

    if args.cmd == "capture":
        model, model_args, *model_kwargs = load_function(args.spec)(**dict(map(parse_arg, args.arg)))
        keep = () if args.decompose_all else tuple(args.keep_op or DEFAULT_KEEP_OPS)
        graphs = [capture(model, model_args, model_kwargs[0] if model_kwargs else None, keep, args.name)]
        out = args.out or Path(graphs[0]["model"]["name"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_name(out.name + ".json").write_text(json.dumps(graphs[0]))
        print(f"wrote {out}.json  ({len(graphs[0]['ops'])} ops, {len(graphs[0]['modules'])} modules)")
        slugs = None
    else:
        graphs = [json.loads(p.read_text()) for p in args.graph]
        out = (args.out or args.graph[0]).with_suffix("")
        slugs = [p.stem for p in args.graph]
    out.with_name(out.name + ".html").write_text(render_html(graphs, slugs))
    print(f"wrote {out}.html")
    if args.model_explorer:
        out.with_name(out.name + ".model_explorer.json").write_text(json.dumps(to_model_explorer(graphs[0])))
        print(f"wrote {out}.model_explorer.json")
