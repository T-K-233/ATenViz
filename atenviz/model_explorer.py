"""Convert an ATenViz graph to the graph JSON of Google AI Edge Model Explorer.

Model Explorer (https://github.com/google-ai-edge/model-explorer) opens this JSON file directly. Each
module becomes a layer (a ``/``-separated namespace), and the module totals are layer attributes.
Constant-foldable ops and weights are not in the graph, as in the viewer.
"""


def to_model_explorer(graph: dict) -> dict:
    tensors, modules = graph["tensors"], graph["modules"]
    ns = [""] * len(modules)
    for m in modules[1:]:  # a parent comes before its children
        ns[m["id"]] = f"{ns[m['parent']]}/{m['name']}".lstrip("/")
    shape = lambda t: f"{tensors[t]['dtype']}{tensors[t]['shape']}".replace(" ", "")
    kv = lambda d: [{"key": k, "value": v} for k, v in d.items()]
    nodes, slot = [], {}  # slot: tensor id -> (node id, output index)

    def edges(ts: list[int]) -> list[dict]:
        return [{"sourceNodeId": slot[t][0], "sourceNodeOutputId": str(slot[t][1]), "targetNodeInputId": str(j)}
                for j, t in enumerate(ts) if t in slot]

    for i, t in enumerate(graph["inputs"]):
        slot[t] = (f"input_{i}", 0)
        nodes.append({"id": f"input_{i}", "label": "Input", "namespace": "", "attrs": kv({"name": tensors[t]["name"]}),
                      "outputsMetadata": [{"id": "0", "attrs": kv({"tensor_shape": shape(t)})}]})
    for op in graph["ops"]:
        if op["const"]:
            continue
        nid = f"op_{op['id']}"
        attrs = {"target": op["target"], "dtype": op["dtype"], "flops_matmul": str(op["flops"]["matmul"]),
                 "flops_nonmatmul": str(op["flops"]["nonmatmul"]), "weight_bytes": str(op["weight_bytes"]),
                 "source": op["source"] or "", **{k: str(v) for k, v in op["attrs"].items()},
                 **{f"weight:{tensors[t]['source'] or tensors[t]['name']}": shape(t) for t in op["weight_reads"]}}
        nodes.append({"id": nid, "label": op["target"].removeprefix("aten.").removesuffix(".default"),
                      "namespace": ns[op["module"]], "attrs": kv(attrs), "incomingEdges": edges(op["inputs"]),
                      "outputsMetadata": [
                          {"id": str(j), "attrs": kv({"tensor_shape": shape(t), "__tensor_tag": tensors[t]["name"]})}
                          for j, t in enumerate(op["outputs"])]})
        slot.update({t: (nid, j) for j, t in enumerate(op["outputs"])})
    for i, t in enumerate(graph["outputs"]):
        nodes.append({"id": f"output_{i}", "label": "Output", "namespace": "", "incomingEdges": edges([t])})

    def layer(m: dict) -> dict:
        s = m["stats"]
        attrs = {"class": m["class"], "path": m["path"], "ops": s["ops"], "weight_bytes_read": s["weight_bytes"],
                 "weight_bytes_unique": s["weight_footprint"]}
        for cat in ("matmul", "nonmatmul"):
            attrs[f"flops_{cat}"] = sum(s["flops"][cat].values())
            attrs.update({f"flops_{cat}[{k}]": n for k, n in s["flops"][cat].items()})
        return {k: str(v) for k, v in attrs.items()}

    return {"label": graph["model"]["name"], "graphs": [{
        "id": graph["model"]["name"], "nodes": nodes, "groupNodeAttributes": {ns[m["id"]]: layer(m) for m in modules},
        "layoutConfigs": {"keepLayersWithASingleChild": True}}]}
