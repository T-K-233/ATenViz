"""Off-chip memory traffic and operational intensity of an ATenViz graph.

The traffic of a graph is not a property of the graph alone. It depends on which ops run as one
kernel and on how much on-chip memory holds data between kernels. ``traffic`` counts it for stated
values of both:

- ``fusion``: the kernels.

  - ``"none"``: each op that is not a view is a kernel (eager PyTorch).
  - ``"vector"``: the connected non-matmul ops between two matmuls are one kernel. Two ops fuse only
    if no matmul is on a path between them, so that the kernels stay acyclic.
  - ``"epilogue"`` (the default): also, a matmul and the vector kernel that is the only consumer of its
    output are one kernel (a bias, an activation or a residual add in the matmul's epilogue).

  A matmul op (``flops.matmul > 0``) is one kernel with its epilogue at most. An attention op kept by
  the capture (``scaled_dot_product_attention``) is one fused kernel, so its scores never leave the chip.
- ``capacity``: the bytes of on-chip memory. It holds whole tensors: weights, inputs and kernel
  outputs. A kernel reads each input that is not on chip from DRAM. Its outputs go on chip, and when
  the memory is full, the tensor whose next use is the latest leaves first (Belady's rule, here a
  heuristic because the tensors differ in size). A tensor larger than ``capacity`` streams: its reads
  and writes go to DRAM.

  - ``capacity=0``: each kernel reads its inputs from DRAM and writes its outputs to DRAM.
  - ``capacity=math.inf``: the compulsory traffic: each weight and input is read once, and each output
    is written once.
- ``writeback``: which outputs that leave the chip are written to DRAM.

  - ``"live"`` (a scratchpad that software manages): only a tensor that a later kernel reads, or a
    graph output.
  - ``"all"`` (a write-back cache): every output, also a dead one. A cache cannot tell that no kernel
    reads the data again.

Other rules:

- A view reads at most the bytes of its storage. A ``clone`` of a broadcast view (the keys and values
  that grouped-query attention repeats for each head) is a view: attention kernels read the shared heads. A constant op (``const``) is folded before the run: its
  output is a weight in DRAM.
- A gather from a weight (an embedding lookup, the experts that a router selects) is a view of the
  weight: the op that uses the rows reads them from the weight, at most the bytes of the weight. Thus a
  trace that gathers one copy of the experts for each token (``batched_mm``) reads each expert at most
  once in a kernel, as a grouped matmul does.
- Output assembly: a ``cat`` or ``stack`` that makes a graph output (the caches of all layers in one
  tensor) moves no data: its inputs are graph outputs, written in place.
- State update: a ``cat``, ``index_put`` or ``*_scatter`` whose first input is a graph input and whose
  output is a graph output (a key-value cache, a recurrent state) updates the state in place. It writes
  only its other inputs to DRAM.
- ``act_dtype`` and ``weight_dtype`` count the floating-point tensors in the deployed dtypes, not in the
  dtypes of the capture. A cast that the dtypes make an identity (``_to_copy`` from fp32 to bf16 when
  the activations are bf16) is a view.
- FLOPs are the ``flops`` of the non-constant ops. The intensity is FLOPs per byte of reads and writes.
"""

import bisect
import fnmatch
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

FLOAT = {"fp64", "fp32", "fp16", "bf16", "fp8e4m3", "fp8e5m2"}
BITS = {"fp64": 64, "fp32": 32, "fp16": 16, "bf16": 16, "fp8e4m3": 8, "fp8e5m2": 8, "i64": 64, "i32": 32,
        "i16": 16, "i8": 8, "u8": 8, "bool": 8, "c64": 64, "c128": 128,
        **{f"{s}{b}": b for s in "iu" for b in range(1, 8)}}
GATHER = {"embedding", "index_select", "index", "gather", "take"}
STATE_UPDATE = {"cat", "index_put", "slice_scatter", "select_scatter", "scatter", "index_copy"}


def load(path) -> dict:
    """Read a graph from a ``.json`` or ``.json.gz`` file."""
    path = Path(path)
    with (gzip.open(path, "rt") if path.suffix == ".gz" else open(path)) as f:
        return json.load(f)


def _packet(op: dict) -> str:
    return op["target"].split(".")[1]


def traffic(graph: dict, capacity: float = math.inf, fusion: str = "epilogue", writeback: str = "live",
            act_dtype: str | None = None, weight_dtype=None) -> dict:
    """Return the DRAM traffic of one run of ``graph`` (see the module docstring).

    ``act_dtype`` is one dtype for every floating-point tensor that is not a weight (inputs, outputs and
    activations), a dict from a glob on the tensor name to a dtype (the first match wins), or a function
    ``(tensor) -> dtype or None`` that gets the tensor's dict.
    ``weight_dtype`` is one dtype name for every floating-point weight, a dict from a glob on the
    weight name to a dtype name (the first match wins; no match keeps the dtype of the capture), or a
    function ``(weight name, op) -> dtype name or None`` for a weight whose dtype depends on the op that
    reads it. A weight in two dtypes is two copies in memory (an FP8 copy for one pass, a bf16 copy for
    another). A dtype can also be a number of bits per element, for a format with group scales (4.5 for
    4-bit weights with a 16-bit scale and bias per 64).

    The result has ``reads``, ``writes`` and ``bytes`` (their sum), ``flops``, ``intensity`` (FLOPs per
    byte), ``weight_reads`` (the part of ``reads`` from weights) and ``kernels``: one dict for each
    kernel in run order, with its ``ops``, ``module`` (of its first matmul, else its first op),
    ``reads``, ``writes`` and ``flops``.
    """
    if fusion not in ("none", "vector", "epilogue") or writeback not in ("live", "all"):
        raise ValueError(f"fusion {fusion!r} or writeback {writeback!r}")
    T, O = graph["tensors"], graph["ops"]

    def dtype(t: dict, op: dict | None = None) -> str:
        d = t["dtype"]
        if d not in FLOAT:
            return d
        if t["const"]:
            name = T[t["storage"]]["name"] if t.get("storage") is not None else t["name"]
            if isinstance(weight_dtype, str):
                return weight_dtype
            if callable(weight_dtype):
                return weight_dtype(name, op) or d
            return next((v for k, v in (weight_dtype or {}).items() if fnmatch.fnmatchcase(name, k)), d)
        if isinstance(act_dtype, dict):
            return next((v for k, v in act_dtype.items() if fnmatch.fnmatchcase(t["name"], k)), d)
        if callable(act_dtype):
            return act_dtype(t) or d
        return act_dtype or d

    def nbytes(t: dict, d) -> int:
        return math.ceil(math.prod(t["shape"]) * (d if isinstance(d, (int, float)) else BITS[d]) / 8)

    size = {t["id"]: nbytes(t, dtype(t)) for t in T if not t["const"]}  # by root; a weight's key is (root, dtype)

    # The storage (root) of each tensor, and the op that writes it.
    root = list(range(len(T)))
    for t in T:
        if t["const"] and t.get("storage") is not None:
            root[t["id"]] = t["storage"]
    alias = [op["view"] or (_packet(op) == "_to_copy" and op["inputs"] and
                            dtype(T[op["inputs"][0]]) == dtype(T[op["outputs"][0]])) for op in O]
    weight_of, index_of = {}, {}  # a gathered view of a weight -> the weight tensor, the index tensors
    for op in O:
        if op["const"] or not op["inputs"]:
            continue
        src = op["inputs"][0]
        if _packet(op) == "clone" and math.prod(T[src]["shape"]) > math.prod(T[root[src]]["shape"]):
            alias[op["id"]] = True  # a broadcast made contiguous
        if _packet(op) in GATHER and T[src]["const"]:
            alias[op["id"]] = True
            weight_of.update((t, src) for t in op["outputs"])
            index_of.update((t, op["inputs"][1:]) for t in op["outputs"])
        elif alias[op["id"]] and src in weight_of:
            weight_of.update((t, weight_of[src]) for t in op["outputs"])
            index_of.update((t, index_of[src]) for t in op["outputs"])
        if alias[op["id"]]:
            for t in op["outputs"]:
                root[t] = root[src]
    in_roots = {root[t] for t in graph["inputs"]}
    out_roots = {root[t] for t in graph["outputs"]}
    assemble = set()
    for op in reversed(O):
        if (not op["const"] and _packet(op) in ("cat", "stack") and op["id"] not in assemble
                and all(root[t] in out_roots for t in op["outputs"])):
            assemble.add(op["id"])
            out_roots.update(root[t] for t in op["inputs"])
    run = [op for op in O if not (op["const"] or alias[op["id"]] or op["id"] in assemble)]
    producer = {}
    for op in run:
        for t in op["outputs"]:
            producer[root[t]] = op["id"]
    update = {op["id"] for op in run if _packet(op) in STATE_UPDATE and op["inputs"]
              and root[op["inputs"][0]] in in_roots and any(root[t] in out_roots for t in op["outputs"])}

    # Kernels: union-find over the ops.
    parent = {op["id"]: op["id"] for op in run}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    anchor = {op["id"]: op["flops"]["matmul"] > 0 for op in run}
    depth = {}
    for op in run:  # matmuls on the longest path from the graph inputs
        ps = [producer[root[t]] for t in op["inputs"] if root[t] in producer]
        depth[op["id"]] = max((depth[p] + anchor[p] for p in ps), default=0)
    if fusion != "none":
        for op in run:
            o = op["id"]
            if anchor[o] or o in update:
                continue
            for t in op["inputs"]:
                p = producer.get(root[t])
                if p is not None and not anchor[p] and p not in update and depth[p] == depth[o]:
                    parent[find(p)] = find(o)
    if fusion == "epilogue":
        consumers = defaultdict(set)
        for op in run:
            for t in op["inputs"]:
                if root[t] in producer:
                    consumers[producer[root[t]]].add(op["id"])
        taken = set()
        for op in reversed(run):
            a = op["id"]
            cs = consumers[a]
            groups = {find(c) for c in cs}
            if (anchor[a] and len(groups) == 1 and not any(anchor[c] or c in update for c in cs)
                    and all(depth[c] == depth[a] + 1 for c in cs) and not any(root[t] in out_roots for t in op["outputs"])):
                g = groups.pop()
                if g not in taken:
                    taken.add(g)
                    parent[find(a)] = g
    members = defaultdict(list)
    for op in run:
        members[find(op["id"])].append(op["id"])

    # Each kernel's inputs (root -> bytes read) and outputs (roots written), in a topological order.
    kin, kout, kflops = {}, {}, {}
    for k, ops in members.items():
        inside = {root[t] for o in ops for t in O[o]["outputs"]}
        views = defaultdict(set)
        for o in ops:
            for j, t in enumerate(O[o]["inputs"] + [x for t in O[o]["inputs"] for x in index_of.get(t, ())]):
                r = root[t]
                if r in inside or (j == 0 and o in update):
                    continue
                if T[t]["const"] or t in weight_of:
                    d = dtype(T[weight_of.get(t, t)], O[o])
                    r = (r, d)
                    size[r] = nbytes(T[r[0]], d)
                    views[r].add((t, nbytes(T[t], d)))
                else:
                    views[r].add((t, size[t] if t in size else nbytes(T[t], dtype(T[t]))))
        kin[k] = {}
        for r, ts in views.items():
            b = sum(b for _, b in ts)
            if b < size[r]:  # a part of the storage: on chip as its own entry, for the kernels that read these views
                r = ("part", r, frozenset(t for t, _ in ts))
                size[r] = b
            kin[k][r] = min(size[r], b)
        kout[k] = inside
        kflops[k] = sum(O[o]["flops"]["matmul"] + O[o]["flops"]["nonmatmul"] for o in ops)
    read_by = defaultdict(set)
    for k in members:
        for r in kin[k]:
            read_by[r].add(k)
    for k in members:  # a tensor that only its own kernel reads stays in registers
        kout[k] = {r for r in kout[k] if r in out_roots or read_by[r] - {k}}
    made_by = {r: k for k in members for r in kout[k]}
    deps = {k: {made_by[r] for r in kin[k] if r in made_by} - {k} for k in members}
    order, done, ready = [], set(), sorted(k for k in members if not deps[k])
    waiting = {k: set(d) for k, d in deps.items() if d}
    users = defaultdict(list)
    for k, d in deps.items():
        for x in d:
            users[x].append(k)
    while ready:
        k = ready.pop(0)
        order.append(k)
        done.add(k)
        for u in users[k]:
            waiting[u].discard(k)
            if not waiting[u]:
                del waiting[u]
                bisect.insort(ready, u)
    if waiting:
        raise ValueError("atenviz.traffic: the fused kernels form a cycle")

    uses = defaultdict(list)  # root -> run positions of the kernels that read it
    for i, k in enumerate(order):
        for r in kin[k]:
            uses[r].append(i)

    def next_use(r: int, i: int) -> float:
        u = uses[r]
        j = bisect.bisect_right(u, i)
        return u[j] if j < len(u) else math.inf

    resident = {}  # root -> dirty
    used = 0
    reads = writes = weight_reads = 0
    rows = []
    for i, k in enumerate(order):
        rd = wr = 0
        for r, b in kin[k].items():
            if r in resident:
                continue
            rd += b
            if isinstance(r, tuple) and (r[0] != "part" or isinstance(r[1], tuple)):
                weight_reads += b
            if b == size[r] and size[r] <= capacity:
                resident[r] = False
                used += size[r]
        for o in members[k]:
            if o in update:  # the new values go into the state in DRAM
                wr += sum(nbytes(T[t], dtype(T[t], O[o])) for t in O[o]["inputs"][1:])
        for r in kout[k]:
            if producer[r] in update:
                continue
            live = r in out_roots or next_use(r, i) < math.inf
            if size[r] <= capacity:
                resident[r] = True
                used += size[r]
            elif live or writeback == "all":
                wr += size[r]
        while used > capacity:
            r = max(resident, key=lambda x: (next_use(x, i), size[x]))
            if resident.pop(r) and (writeback == "all" or r in out_roots or next_use(r, i) < math.inf):
                wr += size[r]
            used -= size[r]
        reads += rd
        writes += wr
        first = next((o for o in members[k] if anchor[o]), members[k][0])
        rows.append({"ops": members[k], "module": O[first]["module"], "reads": rd, "writes": wr, "flops": kflops[k]})
    tail = sum(size[r] for r, dirty in resident.items() if dirty and r in out_roots)
    writes += tail
    if rows:
        rows[-1]["writes"] += tail
    flops = sum(kflops.values())
    total = reads + writes
    return {"capacity": capacity, "fusion": fusion, "writeback": writeback, "reads": reads, "writes": writes,
            "bytes": total, "weight_reads": weight_reads, "flops": flops,
            "intensity": flops / total if total else math.inf, "kernels": rows}


def by_module(graph: dict, result: dict, depth: int = 1) -> dict[str, dict]:
    """Sum the kernels of ``result`` by their module ``depth`` levels below the root (by module path)."""
    mods = graph["modules"]
    out = defaultdict(lambda: {"reads": 0, "writes": 0, "flops": 0, "kernels": 0})
    for row in result["kernels"]:
        chain, m = [], row["module"]
        while m is not None:
            chain.append(m)
            m = mods[m]["parent"]
        chain.reverse()
        m = chain[min(depth, len(chain) - 1)]
        s = out[mods[m]["path"] or mods[m]["name"]]
        for f in ("reads", "writes", "flops"):
            s[f] += row[f]
        s["kernels"] += 1
    return dict(out)
