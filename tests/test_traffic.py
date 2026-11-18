import math

import pytest
import torch
from torch import nn

from atenviz import capture
from atenviz.traffic import traffic
from atenviz.cli import main

from test_capture import Net, ids


@pytest.fixture(scope="module")
def graph():
    return capture(Net().eval(), (ids(),))


def test_compulsory_traffic(graph):
    # Unlimited on-chip memory: the input, each weight once (the shared Linear too), the selected
    # embedding rows and the folded scale vector are read; the output is written once.
    net = Net()
    params = sum(p.numel() * 4 for n, p in net.named_parameters() if not n.startswith("emb"))
    r = traffic(graph, math.inf)
    assert r["reads"] == 2 * 8 * 8 + 2 * 8 * 16 * 4 + params + 16 * 4
    assert r["weight_reads"] == r["reads"] - 2 * 8 * 8
    assert r["writes"] == 2 * 8 * 16 * 4


def test_traffic_falls_with_capacity_and_fusion(graph):
    sizes = [traffic(graph, c)["bytes"] for c in (0, 4096, math.inf)]
    assert sizes[0] > sizes[1] >= sizes[2]
    fused = [traffic(graph, 0, f) for f in ("none", "vector", "epilogue")]
    assert fused[0]["bytes"] > fused[1]["bytes"] > fused[2]["bytes"]
    assert len(fused[0]["kernels"]) > len(fused[1]["kernels"]) > len(fused[2]["kernels"])
    assert all(r["flops"] == fused[0]["flops"] for r in fused)
    # A write-back cache also writes the outputs that no kernel reads again.
    assert traffic(graph, 4096, "none", "all")["writes"] >= traffic(graph, 4096, "none", "live")["writes"]


def test_deployed_dtypes(graph):
    bf16 = traffic(graph, math.inf, act_dtype="bf16", weight_dtype="bf16")
    fp32 = traffic(graph, math.inf)
    assert bf16["bytes"] == pytest.approx(fp32["bytes"] / 2, abs=2 * 8 * 8)  # the int64 ids keep their size
    half = traffic(graph, math.inf, weight_dtype={"blocks.*": "bf16"})
    assert fp32["weight_reads"] > half["weight_reads"] > bf16["weight_reads"]
    # A weight that two ops read in different dtypes is two copies.
    linear = (16 * 16 + 16) * 4
    first = {}

    def per_op(name, op):
        if name.startswith("shared"):
            first.setdefault(name, op["id"])
            return "fp32" if op["id"] == first[name] else "bf16"
        return None

    assert traffic(graph, math.inf, weight_dtype=per_op)["weight_reads"] == fp32["weight_reads"] + linear // 2


class Cache(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.k = nn.Linear(d, d, bias=False)

    def forward(self, x, cache):
        cache = torch.cat([cache, self.k(x)], 1)
        return x @ cache.transpose(1, 2), cache


def test_state_update_writes_only_the_new_values():
    g = capture(Cache().eval(), (torch.randn(1, 1, 8), torch.randn(1, 100, 8)))
    r = traffic(g, math.inf)
    assert r["writes"] == 1 * 101 * 4 + 1 * 8 * 4  # the scores, and one new row of the cache
    assert r["reads"] == 8 * 4 + 64 * 4 + 101 * 8 * 4  # x, the weight, the cache with its new row


class Caches(nn.Module):  # two layers' caches in one tensor, as a decode step returns them
    def __init__(self, d=8):
        super().__init__()
        self.k = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(2)])

    def forward(self, x, caches):
        new = [torch.cat([caches[i], self.k[i](x)], 1) for i in range(2)]
        return sum(x @ c.transpose(1, 2) for c in new), torch.stack(new)


def test_stacked_state_updates_write_only_the_new_values():
    g = capture(Caches().eval(), (torch.randn(1, 1, 8), torch.randn(2, 1, 100, 8)))
    r = traffic(g, math.inf)
    assert r["writes"] == 1 * 101 * 4 + 2 * 8 * 4


class Gqa(nn.Module):
    def forward(self, q, k, v):  # two key-value heads repeated for eight query heads, as transformers does
        k, v = (x[:, :, None].expand(1, 2, 4, 64, 16).reshape(1, 8, 64, 16) for x in (k, v))
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def test_repeated_heads_are_read_once():
    g = capture(Gqa(), (torch.randn(1, 8, 64, 16), torch.randn(1, 2, 64, 16), torch.randn(1, 2, 64, 16)))
    r = traffic(g, 0)
    assert r["reads"] == (8 + 2 + 2) * 64 * 16 * 4 and len(r["kernels"]) == 1


class Experts(nn.Module):
    def __init__(self, e=16, d=8):
        super().__init__()
        self.w = nn.Parameter(torch.randn(e, d, d))

    def forward(self, x, idx):  # one gathered copy of the experts for each token, as batched_mm traces
        return torch.bmm(x[:, None], self.w[idx])[:, 0]


def test_gathered_weights_are_read_at_most_once():
    w = 16 * 8 * 8 * 4
    many = capture(Experts().eval(), (torch.randn(64, 8), torch.arange(64) % 16))
    assert traffic(many, 0)["weight_reads"] == w
    few = capture(Experts().eval(), (torch.randn(2, 8), torch.tensor([3, 5])))
    assert traffic(few, 0)["weight_reads"] == 2 * 8 * 8 * 4


def test_cli_traffic(tmp_path, capsys, graph):
    import json
    path = tmp_path / "g.json"
    path.write_text(json.dumps(graph))
    main(["traffic", str(path), "-c", "0", "-c", "inf", "--act-dtype", "bf16", "--weight-dtype", "blocks.*=bf16",
          "--weight-dtype", "4.5", "--by-module", "1"])
    out = capsys.readouterr().out
    assert out.count("capacity") == 2 and "blocks" in out
