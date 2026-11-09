import base64
import gzip
import json
import re
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from model_viewer import capture, render_html, to_model_explorer
from model_viewer.cli import main, parse_arg


class Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, -1)
        return self.o(F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])[:, 0])


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.attn = Attn(d)

    def forward(self, x):
        return torch.relu(x + self.attn(self.ln(x)))


class Net(nn.Module):
    def __init__(self, d=16, vocab=50):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d), Block(d)])
        self.shared = nn.Linear(d, d)
        self.register_buffer("scale", torch.arange(d, dtype=torch.float32))

    def forward(self, ids):
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x)
        x = self.shared(self.shared(x))
        return x * (self.scale * 2 + 1)  # the scale expression only reads a buffer: constant-foldable


def ids():
    return torch.randint(0, 50, (2, 8))


@pytest.fixture(scope="module")
def graph():
    return capture(Net().eval(), (ids(),))


def modules_by_path(g):
    return {(m["path"], m["name"]): m for m in g["modules"]}


def total(stats, cat):
    return sum(stats["flops"][cat].values())


def test_matmul_flops_match_flop_counter_mode(graph):
    model = Net().eval()
    with FlopCounterMode(display=False) as counter:
        model(ids())
    assert total(graph["modules"][0]["stats"], "matmul") == counter.get_total_flops()


def test_hierarchy_follows_module_calls(graph):
    mods = modules_by_path(graph)
    assert mods[("blocks.0.attn.qkv", "qkv")]["parent"] == mods[("blocks.0.attn", "attn")]["id"]
    # A module called twice gives two calls.
    assert ("shared", "shared") in mods and ("shared", "shared@1") in mods
    # Every op belongs to a module call, and module stats add up to the root.
    root = graph["modules"][0]["stats"]
    children = [graph["modules"][c]["stats"] for c in graph["modules"][0]["children"]]
    direct = [o for o in graph["ops"] if o["module"] == 0 and not o["const"]]
    assert root["ops"] == sum(c["ops"] for c in children) + len(direct)


class Layer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.inp = nn.Linear(d, d)
        self.out = nn.Linear(d, d)


class Loop(nn.Module):
    """Calls the parts of its layers from its own loop, as SmolVLA does."""

    def __init__(self, d=8):
        super().__init__()
        self.layers = nn.ModuleList([Layer(d), Layer(d)])

    def forward(self, x):
        for layer in self.layers:
            h = layer.inp(x)
            h = torch.tanh(h) * 2  # between two calls of the same layer: belongs to the layer
            x = x + layer.out(h)
        return x


def test_owner_modules_for_calls_that_skip_levels():
    g = capture(Loop(), (torch.randn(2, 8),))
    mods = {m["path"]: m for m in g["modules"]}
    assert mods["layers"]["class"].endswith("ModuleList") and mods["layers"]["call"] is False
    assert mods["layers.1"]["class"].endswith("Layer") and mods["layers.1"]["parent"] == mods["layers"]["id"]
    assert mods["layers.1.out"]["parent"] == mods["layers.1"]["id"]
    targets = {o["target"] for o in g["ops"] if o["module"] == mods["layers.0"]["id"]}
    assert {"aten.tanh.default", "aten.mul.Tensor"} <= targets
    # Parents come before children, and the residual add after the last call stays outside.
    assert all(m["parent"] is None or m["parent"] < m["id"] for m in g["modules"])
    assert any(o["target"] == "aten.add.Tensor" and o["module"] == 0 for o in g["ops"])


class Sampler(nn.Module):
    """A flow-matching style loop in plain methods, as VLA policies write it."""

    def __init__(self, d=8):
        super().__init__()
        self.embed = nn.Linear(d, d)
        self.step = nn.Linear(d, d)

    def prefix(self, x):
        return torch.relu(self.embed(x)) * 2

    def denoise(self, h, t):
        noise = torch.randn_like(h)  # random: never constant-foldable
        return self.step(h + noise * t) - h

    @torch.no_grad()
    def forward(self, x):
        h = self.prefix(x)
        for i in range(3):
            h = h + 0.1 * self.denoise(h, 1.0 - i / 3)
        return h


def test_function_scopes():
    g = capture(Sampler(), (torch.randn(2, 8),))
    M, O = g["modules"], g["ops"]
    scopes = [m for m in M if m["kind"] == "function"]
    assert [m["name"] for m in scopes] == ["prefix", "denoise", "denoise@1", "denoise@2"]
    assert all(m["parent"] == 0 and m["file"].endswith("test_capture.py") for m in scopes)
    step = next(m for m in M if m["name"] == "step@1")
    assert M[step["parent"]]["name"] == "denoise@1"
    # The Euler updates between the steps stay in the root, and the noise is not folded.
    assert any(o["target"] == "aten.add.Tensor" and o["module"] == 0 for o in O)
    rand = [o for o in O if "rand" in o["target"]]
    assert len(rand) == 3 and not any(o["const"] for o in rand)
    assert all(o["source"] and "test_capture.py" in o["source"] for o in rand)


def test_cli_args():
    assert parse_arg("checkpoint=lerobot/pi0_base") == ("checkpoint", "lerobot/pi0_base")
    assert parse_arg("num_inference_steps=5") == ("num_inference_steps", 5)
    assert parse_arg("flag=True") == ("flag", True)


def test_module_inputs_and_outputs(graph):
    t = graph["tensors"]
    block = modules_by_path(graph)[("blocks.0", "0")]
    assert [t[i]["shape"] for i in block["inputs"]] == [[2, 8, 16]]
    assert [t[i]["shape"] for i in block["outputs"]] == [[2, 8, 16]]
    nxt = modules_by_path(graph)[("blocks.1", "1")]
    assert block["outputs"] == nxt["inputs"]
    assert all(not t[i]["const"] for i in block["inputs"] + block["outputs"])


def test_weight_traffic(graph):
    mods = modules_by_path(graph)
    shared = [mods[("shared", "shared")], mods[("shared", "shared@1")]]
    linear_bytes = (16 * 16 + 16) * 4
    assert [m["stats"]["weight_bytes"] for m in shared] == [linear_bytes, linear_bytes]
    # An embedding lookup reads only the selected rows.
    table, rows = 50 * 16 * 4, 2 * 8 * 16 * 4
    assert mods[("emb", "emb")]["stats"]["weight_bytes"] == rows
    assert mods[("emb", "emb")]["stats"]["weight_footprint"] == table
    # The footprint counts each parameter once, plus the folded scale vector. The traffic counts
    # the shared Linear twice and the gathered embedding rows instead of the table.
    params = sum(p.numel() * 4 for p in Net().parameters())
    root = graph["modules"][0]["stats"]
    assert root["weight_footprint"] == params + 16 * 4
    assert root["weight_bytes"] == params - table + rows + linear_bytes + 16 * 4


def test_constant_folding(graph):
    ops = graph["ops"]
    const = [o for o in ops if o["const"]]
    assert {o["target"] for o in const} >= {"aten.mul.Tensor", "aten.add.Tensor"}
    final = ops[graph["tensors"][graph["outputs"][0]]["producer"]]
    assert final["weight_bytes"] == 16 * 4  # it reads the folded scale vector


def test_sdpa_kept_by_default_and_decomposed_on_request(graph):
    targets = {o["target"] for o in graph["ops"]}
    assert "aten.scaled_dot_product_attention.default" in targets
    full = capture(Net().eval(), (ids(),), keep_ops=())
    targets = {o["target"] for o in full["ops"]}
    assert "aten.scaled_dot_product_attention.default" not in targets
    assert "aten.bmm.default" in targets
    assert any(m["kind"] == "composite" for m in full["modules"])
    assert total(full["modules"][0]["stats"], "matmul") == total(graph["modules"][0]["stats"], "matmul")


def test_dtype_pairs():
    model = nn.Sequential(nn.Linear(16, 32), nn.GELU()).to(torch.bfloat16)
    g = capture(model, (torch.randn(4, 16, dtype=torch.bfloat16),))
    stats = g["modules"][0]["stats"]
    assert set(stats["flops"]["matmul"]) == {"bf16×bf16→bf16"}
    assert stats["flops"]["matmul"]["bf16×bf16→bf16"] == 2 * 4 * 16 * 32
    assert set(stats["flops"]["nonmatmul"]) == {"bf16×bf16→bf16", "bf16→bf16"}  # bias add, gelu


def test_view_ops_cost_nothing(graph):
    views = [o for o in graph["ops"] if o["view"]]
    assert views and all(o["flops"] == {"matmul": 0, "nonmatmul": 0} for o in views)


def test_meta_device_model():
    with torch.device("meta"):
        model = nn.Sequential(nn.Linear(1024, 4096), nn.ReLU(), nn.Linear(4096, 1024))
    g = capture(model, (torch.randn(8, 1024, device="meta"),))
    assert total(g["modules"][0]["stats"], "matmul") == 2 * 2 * 8 * 1024 * 4096


def test_render_html_embeds_graph_safely(graph):
    g = json.loads(json.dumps(graph))
    g["model"]["name"] = "</script><b>x"
    html = render_html([g, graph], ["a", "b"])
    assert "</script><b>" not in html
    blobs = re.findall(r'type="application/gzip-base64" data-slug="(\w+)" data-label="[^"]*">([^<]*)</script>', html)
    assert [slug for slug, _ in blobs] == ["a", "b"]
    assert json.loads(gzip.decompress(base64.b64decode(blobs[0][1])))["model"]["name"] == "</script><b>x"


def test_model_explorer_graph_is_consistent(graph):
    me = to_model_explorer(graph)
    nodes = me["graphs"][0]["nodes"]
    ids_ = {n["id"] for n in nodes}
    assert all(e["sourceNodeId"] in ids_ for n in nodes for e in n.get("incomingEdges", []))
    namespaces = {n["namespace"] for n in nodes}
    assert "blocks/0/attn/qkv" in namespaces
    assert "shared@1" in namespaces
    assert me["graphs"][0]["groupNodeAttributes"]["blocks/0"]["class"].endswith("Block")


def test_cli_capture(tmp_path):
    examples = Path(__file__).parents[1] / "examples"
    main(["capture", f"{examples / 'resnet.py'}:build", "-o", str(tmp_path / "r"), "--model-explorer"])
    g = json.loads((tmp_path / "r.json").read_text())
    assert total(g["modules"][0]["stats"], "matmul") == 3_628_146_688  # ResNet-18, 224 x 224
    assert (tmp_path / "r.html").stat().st_size > 1_000_000
    assert (tmp_path / "r.model_explorer.json").exists()
