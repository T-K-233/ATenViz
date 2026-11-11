"""Capture a PyTorch model as a hierarchical ATen graph.

``capture_program`` runs ``torch.export`` once and lowers the graph to Core ATen. The export traces
with FakeTensors, so a model on the meta device works and no op runs. ``build_graph`` converts the
ExportedProgram to the ATenViz graph: a JSON-ready dict of tensors, ops and module calls. See
docs/ir.md for the format.
"""

import math
import operator
import os
import sys
import sysconfig
import warnings
from collections import defaultdict
from unittest import mock

import torch
import torch.fx.proxy
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind
from torch.utils._pytree import tree_flatten, tree_map

from . import costs

FORMAT = "atenviz.graph/1"

# The Core ATen decomposition of scaled_dot_product_attention is the math reference. It upcasts bf16
# and fp16 attention to fp32, so its dtypes are wrong for hardware.
DEFAULT_KEEP_OPS = ("aten.scaled_dot_product_attention.default",)

_TORCH_DIR = os.path.dirname(torch.__file__) + os.sep
_TORCH_MODULES = os.path.join(_TORCH_DIR, "nn", "modules") + os.sep
_MODULE_PY = _TORCH_MODULES + "module.py"
_OTHER_DIRS = (sysconfig.get_paths()["stdlib"] + os.sep, os.path.dirname(__file__) + os.sep)


def _model_frame(path: str) -> bool:
    """Return True for a stack frame of the model's code: not torch (except the torch.nn modules),
    not the standard library (copy.deepcopy) and not this package."""
    if path.startswith(_TORCH_DIR):
        return path.startswith(_TORCH_MODULES)
    return not path.startswith(_OTHER_DIRS)


def _short_path(path: str) -> str:
    """Return ``path`` relative to the ``sys.path`` entry that holds it, which is its import path
    (``lerobot/policies/pi0/modeling_pi0.py``). The graph then has no local directories."""
    roots = [r for r in (os.path.join(os.path.abspath(p), "") for p in sys.path) if path.startswith(r)]
    return path[len(max(roots, key=len)):] if roots else os.path.basename(path)


class _ModelFrames(set):
    """Stack-trace anchors that accept every frame of the model's code."""

    def __contains__(self, frame) -> bool:
        return _model_frame(frame[0])


def capture_program(model: torch.nn.Module, args: tuple = (), kwargs: dict | None = None,
                    keep_ops: tuple[str, ...] = DEFAULT_KEEP_OPS) -> ExportedProgram:
    """Export ``model`` and lower it to Core ATen, except for the ops in ``keep_ops``.

    The export runs under ``torch.no_grad()``, so a ``@torch.no_grad`` method of the model adds no
    grad-mode subgraph. ``torch.export`` records only the stack frames of ``forward`` methods and of
    registered anchors. The capture makes every frame of the model's code an anchor, so that each op
    gets its own source line and its function scope.
    """
    with torch.no_grad(), mock.patch.object(torch.fx.proxy, "_STACK_TRACE_ANCHORS", _ModelFrames()):
        ep = torch.export.export(model, tuple(args), kwargs or {})
    table = torch.export.default_decompositions()
    for name in keep_ops:
        ns, op, overload = name.split(".")
        table.pop(getattr(getattr(getattr(torch.ops, ns), op), overload), None)
    return ep.run_decompositions(table)


def capture(model: torch.nn.Module, args: tuple = (), kwargs: dict | None = None,
            keep_ops: tuple[str, ...] = DEFAULT_KEEP_OPS, name: str | None = None) -> dict:
    """Export ``model`` with example inputs and return the ATenViz graph."""
    graph = build_graph(capture_program(model, args, kwargs, keep_ops), name or type(model).__name__, model)
    graph["model"]["kept_ops"] = list(keep_ops)
    return graph


def _qualname(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _jsonable(v):
    # JSON has no inf or nan, so a non-finite float becomes "inf", "-inf" or "nan".
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return costs.dtype_name(v) if isinstance(v, torch.dtype) else str(v)


def _size(d) -> tuple[int, str | None]:
    """Return a dimension as an int, and its symbol if it is symbolic.

    A data-dependent size (from a boolean mask or ``nonzero``) has no value at export time. It counts
    as the upper bound of its range, or as 1 if the range has no upper bound.
    """
    if isinstance(d, int):
        return d, None
    node = d.node
    if node.hint is not None:
        return int(node.hint), str(node.expr)
    bounds = {}
    for sym in node.expr.free_symbols:
        try:
            bounds[sym] = int(node.shape_env.var_to_range[sym].upper)
        except (KeyError, TypeError, ValueError, OverflowError):
            bounds[sym] = 1
    return int(node.expr.subs(bounds)), str(node.expr)


def _static(x):
    """Return ``x`` with ``_size`` values for its symbolic sizes, so that the FLOP formulas get ints."""
    if isinstance(x, torch.SymInt):
        return _size(x)[0]
    if isinstance(x, torch.Tensor) and not all(isinstance(d, int) for d in x.shape):
        return torch.empty([_size(d)[0] for d in x.shape], dtype=x.dtype, device="meta")
    return x


def _frames(node: torch.fx.Node) -> list[tuple[str, str, str, str]]:
    """Return the frames of the model's code in the stack trace of ``node``, outermost first, as
    ``(file, function, line, code)``."""
    lines = [s.strip() for s in (node.meta.get("stack_trace") or "").splitlines()]
    frames = []
    for i, head in enumerate(lines):
        if not head.startswith('File "'):
            continue
        path, _, rest = head[len('File "'):].partition('", line ')
        line, _, fn = rest.partition(", in ")
        if fn and _model_frame(path):
            code = lines[i + 1] if i + 1 < len(lines) and not lines[i + 1].startswith('File "') else ""
            frames.append((path, fn, line, code))
    return frames


def _new_module(modules: list[dict], kind: str, name: str, path: str, cls: str, parent: int | None,
                **extra) -> int:
    modules.append({"id": len(modules), "kind": kind, "name": name, "path": path, "class": cls,
                    "parent": parent, **extra})
    return len(modules) - 1


def build_graph(ep: ExportedProgram, name: str = "model", model: torch.nn.Module | None = None) -> dict:
    """Convert an ExportedProgram to the ATenViz graph.

    ``model`` is the exported module. It gives the classes of the modules that own module calls but
    are not called themselves.
    """
    kinds = {InputKind.PARAMETER: "param", InputKind.BUFFER: "buffer", InputKind.CONSTANT_TENSOR: "constant"}
    placeholders = {s.arg.name: (kinds.get(s.kind, "input"), s.target) for s in ep.graph_signature.input_specs}
    tensors, ops, frames, packets, inputs, outputs = [], [], [], [], [], []
    modules: list[dict] = []
    _new_module(modules, "module", name, "", name if model is None else _qualname(type(model)), None)
    module_of = {(): 0}  # nn_module_stack keys from the root to a call -> module id
    ids = {}  # fx node -> tensor id, or a list of ids for a tuple output

    def add_tensor(t: torch.Tensor, tname: str, kind: str, producer: int | None, source: str | None = None) -> int:
        sizes = [_size(d) for d in t.shape]
        shape = [n for n, _ in sizes]
        tensors.append({"id": len(tensors), "name": tname, "shape": shape, "dtype": costs.dtype_name(t.dtype),
                        "bytes": costs.nbytes(math.prod(shape), t.dtype), "kind": kind,
                        "const": kind in ("param", "buffer", "constant"), "producer": producer, "source": source})
        if any(sym for _, sym in sizes):
            tensors[-1]["symbolic_shape"] = [sym or n for n, sym in sizes]
        return len(tensors) - 1

    def output_ids(val, tname: str, producer: int):
        if isinstance(val, torch.Tensor):
            return add_tensor(val, tname, "activation", producer)
        if isinstance(val, (list, tuple)):
            return [output_ids(v, f"{tname}.{i}", producer) for i, v in enumerate(val)]
        return None

    def flat(x) -> list[int]:
        return [i for i in tree_flatten(x)[0] if isinstance(i, int)]

    def example(x):
        """Replace the fx nodes in an argument with their example values."""
        return tree_map(lambda v: _static(v.meta.get("val") if isinstance(v, torch.fx.Node) else v), x)

    def module_id(node: torch.fx.Node) -> int:
        key, mid = (), 0
        for call, (path, cls) in (node.meta.get("nn_module_stack") or {}).items():
            if not path:
                continue  # the root module
            key += (call,)
            if key not in module_of:
                suffix = "@" + call.rsplit("@", 1)[1] if "@" in call else ""
                cls = cls if isinstance(cls, str) else _qualname(cls)
                module_of[key] = _new_module(modules, "module", path.rsplit(".", 1)[-1] + suffix, path, cls, mid)
            mid = module_of[key]
        return mid

    for node in ep.graph.nodes:
        val = node.meta.get("val")
        if node.op == "placeholder":
            kind, target = placeholders.get(node.name, ("input", None))
            if isinstance(val, torch.Tensor):
                ids[node] = add_tensor(val, target or node.name, kind, None, target)
                if kind == "input":
                    inputs.append(ids[node])
            continue
        if node.op == "output":
            outputs = flat([ids.get(a) for a in tree_flatten(node.args)[0] if isinstance(a, torch.fx.Node)])
            continue
        if node.target is operator.getitem:
            src, index = node.args
            ids[node] = ids[src][index] if isinstance(ids.get(src), list) else None
            continue
        if node.op != "call_function" or not any(isinstance(v, torch.Tensor) for v in tree_flatten(val)[0]):
            continue  # assertions and symbolic-int arithmetic
        if not isinstance(node.target, torch._ops.OpOverload):
            warnings.warn(f"atenviz: {node.target} is not an ATen op; its tensors have no producer")
            continue

        op = node.target
        named, attrs, in_ids, in_args = {}, {}, [], []
        for i, arg in enumerate(op._schema.arguments):
            if i < len(node.args):
                raw = node.args[i]
            elif arg.name in node.kwargs:
                raw = node.kwargs[arg.name]
            else:
                continue
            named[arg.name] = example(raw)
            arg_ids = flat([ids.get(n) for n in tree_flatten(raw)[0] if isinstance(n, torch.fx.Node)])
            if arg_ids:
                in_ids += arg_ids
                in_args += [arg.name] * len(arg_ids)
            else:
                attrs[arg.name] = _jsonable(named[arg.name])
        op_id = len(ops)
        ids[node] = output_ids(val, node.name, op_id)
        out_ids = flat(ids[node])
        matmul, nonmatmul = costs.op_flops(op, named, list(example(node.args)), example(dict(node.kwargs)),
                                           tree_map(_static, val))
        # A random op (randn, dropout) gives a new value at each call, so it is never constant-foldable.
        const = all(tensors[t]["const"] for t in in_ids) and torch.Tag.nondeterministic_seeded not in op.tags
        for t in out_ids:
            tensors[t]["const"] = const
        origin = (node.meta.get("from_node") or [None])[0]
        decomposed = origin is not None and str(origin.target) != str(op)
        frames.append(_frames(node))
        packets.append(op.overloadpacket)
        source = None
        if frames[-1]:
            path, fn, line, code = frames[-1][-1]
            source = f"{_short_path(path)}:{line} in {fn}: {code}"
        ops.append({
            "id": op_id, "name": node.name, "target": str(op), "module": module_id(node),
            "inputs": in_ids, "input_args": in_args, "outputs": out_ids, "attrs": attrs,
            "flops": {"matmul": matmul, "nonmatmul": nonmatmul},
            "dtype": costs.dtype_pair(op.overloadpacket, in_args, [tensors[t]["dtype"] for t in in_ids],
                                      [tensors[t]["dtype"] for t in out_ids]),
            "category": "matmul" if matmul else "nonmatmul" if nonmatmul else "none",
            "view": costs.is_view(op), "const": const, "weight_reads": [], "weight_bytes": 0,
            "origin": origin.name if decomposed else None, "origin_target": str(origin.target) if decomposed else None,
            "source": source,
        })

    # Weight traffic: a non-constant op reads each constant input (a weight, or a value that
    # constant-foldable ops make from weights) once. A view of a constant (a transposed or expanded
    # weight) reads at most the bytes of the tensor that it is a view of. A gather (an embedding
    # lookup) reads at most the bytes that it writes.
    for t in tensors:
        p = t["producer"]
        view = p is not None and ops[p]["view"] and ops[p]["inputs"]
        t["storage"] = (tensors[ops[p]["inputs"][0]]["storage"] if view else t["id"]) if t["const"] else None
    for op, packet in zip(ops, packets):
        if op["const"]:
            continue
        op["weight_reads"] = list(dict.fromkeys(t for t in op["inputs"] if tensors[t]["const"]))
        limit = sum(tensors[t]["bytes"] for t in op["outputs"]) if packet in costs.GATHER else math.inf
        op["weight_bytes"] = sum(min(tensors[t]["bytes"], tensors[tensors[t]["storage"]]["bytes"], limit)
                                 for t in op["weight_reads"])
    # The source of a constant that ops make: the parameters, buffers and constants it comes from.
    for t in tensors:
        if not t["const"] or t["producer"] is None:
            continue
        roots, todo, seen = {}, [t["id"]], set()
        while todo:
            x = todo.pop()
            if x not in seen:
                seen.add(x)
                p = tensors[x]["producer"]
                if p is None:
                    roots[tensors[x]["name"]] = None
                else:
                    todo += ops[p]["inputs"]
        t["source"] = ", ".join(roots) or None

    _add_owner_modules(modules, ops, model)
    _add_function_scopes(modules, ops, frames)
    _add_composites(modules, ops)
    modules = _renumber(modules, ops)
    _summarize(modules, tensors, ops, inputs, outputs)
    return {"format": FORMAT, "model": {"name": name, "torch_version": torch.__version__},
            "inputs": inputs, "outputs": outputs, "tensors": tensors, "ops": ops, "modules": modules}


def _tree(modules: list[dict], ops: list[dict]) -> tuple[dict, dict, dict]:
    """Return the ops directly in each module, the first op under each module, and the children of
    each module in execution order."""
    direct, first, children = defaultdict(list), {}, defaultdict(list)
    for op in ops:
        direct[op["module"]].append(op["id"])
        m = op["module"]
        while m is not None and m not in first:
            first[m] = op["id"]
            m = modules[m]["parent"]
    for m in sorted(first, key=first.get):
        if modules[m]["parent"] is not None:
            children[modules[m]["parent"]].append(m)
    return direct, first, children


def _add_owner_modules(modules: list[dict], ops: list[dict], model: torch.nn.Module | None) -> None:
    """Add the modules that own module calls but are not called themselves.

    A model can call the parts of a module directly: SmolVLA runs ``layers[i].self_attn.q_proj`` and
    ``layers[i].mlp`` from its own loop, and nothing calls a ModuleList. Then the path of a call skips
    levels of the path of its caller. This pass adds the skipped levels from the attribute paths.
    Consecutive calls under one path share one instance. An op between two calls goes to the deepest
    instance that holds both calls, so the attention math between ``q_proj`` and ``o_proj`` goes to
    ``self_attn``.
    """
    direct, first, children = _tree(modules, ops)
    count = defaultdict(int)
    for parent in range(len(modules)):  # the modules that this loop adds skip no levels
        base = modules[parent]["path"]
        events = sorted([(first[c], 1, c) for c in children[parent]] + [(o, 0, o) for o in direct[parent]])
        stack, pending = [], []  # the open instances as (path, id); the ops after the last call
        for _, is_call, x in events:
            if not is_call:
                pending.append(x)
                continue
            path = modules[x]["path"]
            under = not base or path.startswith(base + ".")
            parts = (path[len(base) + 1:] if base else path).split(".") if under else []
            chain = [".".join(filter(None, [base, *parts[:i]])) for i in range(1, len(parts))]
            k = 0
            while k < min(len(stack), len(chain)) and stack[k][0] == chain[k]:
                k += 1
            for o in pending:
                ops[o]["module"] = stack[k - 1][1] if k else parent
            pending = []
            del stack[k:]
            for p in chain[k:]:
                try:
                    cls = _qualname(type(model.get_submodule(p)))
                except AttributeError:
                    cls = p.rsplit(".", 1)[-1]
                n = count[p]
                count[p] += 1
                name = p.rsplit(".", 1)[-1] + (f"@{n}" if n else "")
                stack.append((p, _new_module(modules, "module", name, p, cls, stack[-1][1] if stack else parent,
                                             call=False)))
            modules[x]["parent"] = stack[-1][1] if stack else parent


def _add_function_scopes(modules: list[dict], ops: list[dict], frames: list[list]) -> None:
    """Group the items of each module by the Python function calls that ran them.

    A model does much of its work in plain methods: ``embed_prefix``, each ``denoise_step`` of a
    flow-matching loop, ``apply_rotary_pos_emb``. In a module, consecutive items (ops and child
    modules) that ran in one call of a function go into a "function" pseudo-module with its name. The
    frames of the code of a module are the stack frames between the ``_call_impl`` frame that called
    it and the next ``_call_impl`` frame. These frames make no group: the frames that all items of the
    module share, anonymous frames (``<lambda>``) and the ``forward`` of the module itself. A function
    that only calls one other function takes the group of that function. A group needs 2 or more
    items that are not views or constants.
    """
    direct, first, children = _tree(modules, ops)
    members = defaultdict(list)
    for op in ops:
        m = op["module"]
        while m is not None:
            members[m].append(op["id"])
            m = modules[m]["parent"]
    real_chain = {}  # module id -> the called modules from the root to it

    def called(m: dict) -> bool:
        return m["parent"] is None or (m["kind"] == "module" and m.get("call", True))

    def chain_of(mid: int) -> list[int]:
        if mid not in real_chain:
            m = modules[mid]
            base = [] if m["parent"] is None else chain_of(m["parent"])
            real_chain[mid] = base + [mid] if called(m) else base
        return real_chain[mid]

    bounds = [[i for i, f in enumerate(fr) if f[0] == _MODULE_PY and f[1] == "_call_impl"] for fr in frames]

    def region(o: int, r: int) -> tuple:
        """Return the frames that op ``o`` runs through in the code of the called module ``r``."""
        rc = chain_of(ops[o]["module"])
        if r not in rc or len(bounds[o]) < len(rc) - 1:
            return ()
        j, k, b = rc.index(r), len(rc) - 1, bounds[o]
        start = b[len(b) - k + j - 1] + 1 if j else (b[len(b) - k - 1] + 1 if len(b) > k else 0)
        end = b[len(b) - k + j] if j < k else len(frames[o])
        code = [f for f in frames[o][start:end] if f[0] != _MODULE_PY]
        return tuple(f[:2] for f in code[1:] if not f[1].startswith("<"))

    def common(a: tuple, b: tuple) -> tuple:
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        return a[:n]

    def item_chain(kind: str, x: int, r: int) -> tuple:
        if kind == "op" or called(modules[x]):
            return region(x if kind == "op" else members[x][0], r)
        prefix = None
        for o in members[x]:
            prefix = region(o, r) if prefix is None else common(prefix, region(o, r))
            if not prefix:
                break
        return prefix or ()

    count = defaultdict(int)

    def build(parent: int, items: list, chains: list, d: int) -> None:
        i = 0
        while i < len(items):
            if len(chains[i]) <= d:
                i += 1
                continue
            j = i
            while j < len(items) and len(chains[j]) > d and chains[j][d] == chains[i][d]:
                j += 1
            run = range(i, j)
            if sum(kind == "mod" or not (ops[x]["view"] or ops[x]["const"]) for kind, x in items[i:j]) >= 2:
                dd = d
                while all(len(chains[t]) > dd + 1 and chains[t][dd + 1] == chains[i][dd + 1] for t in run):
                    dd += 1
                path, fn = chains[i][dd]
                n = count[(parent, fn)]
                count[(parent, fn)] += 1
                scope = _new_module(modules, "function", fn + (f"@{n}" if n else ""),
                                    f"{modules[parent]['path']}/{fn}".lstrip("/"), fn, parent, file=_short_path(path))
                for kind, x in items[i:j]:
                    if kind == "op":
                        ops[x]["module"] = scope
                    else:
                        modules[x]["parent"] = scope
                build(scope, items[i:j], chains[i:j], dd + 1)
            i = j

    for x in range(len(modules)):  # the groups that this loop adds are not visited
        if modules[x]["kind"] != "module":
            continue
        r = chain_of(x)[-1]
        items = sorted([("op", o) for o in direct[x]] + [("mod", c) for c in children[x]],
                       key=lambda it: it[1] if it[0] == "op" else first[it[1]])
        chains = [item_chain(kind, y, r) for kind, y in items]
        if chains:
            shared = chains[0]
            for c in chains[1:]:
                shared = common(shared, c)
            build(x, items, [c[len(shared):] for c in chains], 0)


def _add_composites(modules: list[dict], ops: list[dict]) -> None:
    """Group the ops that one composite op decomposed into (for example the bmm, softmax and views of
    scaled_dot_product_attention) in a "composite" pseudo-module. A group needs 2 or more ops that
    are not views or constants, and it is not a whole module (a Linear)."""
    direct, _, children = _tree(modules, ops)
    groups = defaultdict(list)
    for op in ops:
        if op["origin"]:
            groups[(op["module"], op["origin"])].append(op)
    for (parent, origin), members in groups.items():
        if sum(not (o["view"] or o["const"]) for o in members) < 2:
            continue
        if len(members) == len(direct[parent]) and not children[parent]:
            continue
        target = members[0]["origin_target"]
        cls = target.rsplit(".", 1)[0] if target.count(".") > 1 else target
        mid = _new_module(modules, "composite", origin, f"{modules[parent]['path']}/{origin}".lstrip("/"), cls, parent)
        for o in members:
            o["module"] = mid


def _renumber(modules: list[dict], ops: list[dict]) -> list[dict]:
    """Return the modules in depth-first order, with children in execution order, and with their
    depths, children and ops."""
    direct, _, children = _tree(modules, ops)
    order, todo = [], [0]
    while todo:
        m = todo.pop()
        order.append(m)
        todo += reversed(children[m])
    new = {old: i for i, old in enumerate(order)}
    out = []
    for old in order:
        m = modules[old]
        parent = None if m["parent"] is None else new[m["parent"]]
        out.append({**m, "id": new[old], "parent": parent, "depth": 0 if parent is None else out[parent]["depth"] + 1,
                    "children": [new[c] for c in children[old]], "ops": direct[old]})
    for op in ops:
        op["module"] = new[op["module"]]
    return out


def _summarize(modules: list[dict], tensors: list[dict], ops: list[dict], inputs: list[int],
               outputs: list[int]) -> None:
    """Set the inputs, outputs, weights and stats of each module.

    The inputs and outputs are the activation tensors that cross the boundary of the module. The
    stats sum the FLOPs (by category and dtype pair) and the weight traffic of the ops under the
    module. Constant-foldable ops do not count: their outputs are weights.
    """
    anc = []
    for m in modules:  # a parent comes before its children
        anc.append([m["id"]] if m["parent"] is None else anc[m["parent"]] + [m["id"]])
    io = [{"inputs": {}, "outputs": {}, "weights": {}} for _ in modules]
    stats = [{"flops": {"matmul": {}, "nonmatmul": {}}, "weight_bytes": 0, "weight_footprint": 0, "ops": 0,
              "const_flops": 0} for _ in modules]
    io[0]["inputs"] = dict.fromkeys(inputs)
    for op in ops:
        consumer = anc[op["module"]]
        if op["const"]:
            for m in consumer:
                stats[m]["const_flops"] += op["flops"]["matmul"] + op["flops"]["nonmatmul"]
            continue
        for m in consumer:
            s = stats[m]
            for cat in ("matmul", "nonmatmul"):
                if op["flops"][cat]:
                    s["flops"][cat][op["dtype"]] = s["flops"][cat].get(op["dtype"], 0) + op["flops"][cat]
            s["weight_bytes"] += op["weight_bytes"]
            s["ops"] += 1
        for t in op["inputs"]:
            if tensors[t]["const"]:
                for m in consumer:
                    io[m]["weights"][t] = None
                continue
            p = tensors[t]["producer"]
            producer = anc[ops[p]["module"]] if p is not None else [0]
            k = 0
            while k < min(len(producer), len(consumer)) and producer[k] == consumer[k]:
                k += 1
            for m in consumer[k:]:
                io[m]["inputs"][t] = None
            for m in producer[k:]:
                io[m]["outputs"][t] = None
    for t in outputs:
        p = tensors[t]["producer"]
        for m in anc[ops[p]["module"]] if p is not None else [0]:
            io[m]["outputs"][t] = None
    for m, mio, s in zip(modules, io, stats):
        m.update({k: list(v) for k, v in mio.items()})
        storages = dict.fromkeys(tensors[t]["storage"] for t in m["weights"])
        s["weight_footprint"] = sum(tensors[st]["bytes"] for st in storages)
        m["stats"] = s
