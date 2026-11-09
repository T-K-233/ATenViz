"use strict";
// Model Viewer: draws a model_viewer graph (docs/ir.md) as nested module boxes.
//
// A module is open (a box with its ops, child modules, inputs and outputs) or closed (a summary
// card). Each open module has its own layered layout (ELK, from top to bottom), with its children at
// their current sizes. The layouts are cached by structure, so repeated blocks (the layers of a
// transformer) get one layout. When you zoom in on a card, it opens in place. When you zoom out of a
// small open module, it closes. The module keeps its top edge and its width on the screen.
(() => {
  const OP_W = 184, OP_H = 46, PORT_W = 196, PORT_H = 24, PAD_TOP = 46, PAD = 18;
  const CARD_W = 248, CARD_H = 92;
  const OPEN_PX = 360;     // screen width at which a card under the cursor opens when you zoom in
  const CLOSE_PX = 150;    // screen width below which an open module closes when you zoom out
  const START_ITEMS = 48;  // the first view opens module levels until it shows this many items
  const ELK_OPTIONS = {
    "elk.algorithm": "layered",
    "elk.direction": "DOWN",
    "elk.edgeRouting": "ORTHOGONAL",
    "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
    "elk.layered.nodePlacement.strategy": "BRANDES_KOEPF",
    "elk.spacing.nodeNode": "22",
    "elk.layered.spacing.nodeNodeBetweenLayers": "30",
    "elk.layered.spacing.edgeNodeBetweenLayers": "14",
    "elk.spacing.edgeNode": "12",
    "elk.spacing.edgeEdge": "8",
    "elk.padding": `[top=${PAD_TOP},left=${PAD},bottom=${PAD},right=${PAD}]`,
  };
  // Network-simplex node placement draws straighter wires, but its time grows fast with the number of
  // long edges (21 s for the 84 items of the SmolVLA expert layers, 0.2 s with Brandes-Koepf). Thus
  // only small modules use it.
  const SMALL_OPTIONS = { "elk.layered.nodePlacement.strategy": "NETWORK_SIMPLEX" };
  // Simpler settings for when a layout fails: elkjs can overflow the stack on large graphs
  // (kieler/elkjs#314). If these also fail, the items go in one column.
  const FALLBACK_OPTIONS = { "elk.layered.layering.strategy": "LONGEST_PATH",
    "elk.layered.crossingMinimization.strategy": "INTERACTIVE", "elk.edgeRouting": "POLYLINE" };
  const TEXT_FONT = '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif';
  const MONO_FONT = '"SF Mono", ui-monospace, Menlo, Monaco, Consolas, monospace';

  // ---------- formatting ----------
  function scaled(n, base, units) {
    let i = 0;
    while (Math.abs(n) >= base && i < units.length - 1) { n /= base; i++; }
    return (i ? n.toPrecision(3) : String(Math.round(n))) + " " + units[i];
  }
  const si = (n) => scaled(n, 1000, ["", "K", "M", "G", "T", "P", "E"]);
  const fmtFlops = (n) => si(n) + "FLOP";
  const fmtBytes = (n) => scaled(n, 1024, ["B", "KiB", "MiB", "GiB", "TiB"]);
  const shapeStr = (s) => "[" + s.join(",") + "]";
  const opShort = (target) => target.replace(/^aten\./, "").replace(/\.default$/, "");
  const classShort = (c) => c.split(".").pop();
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const sum = (o) => Object.values(o).reduce((a, b) => a + b, 0);
  const totalFlops = (stats) => sum(stats.flops.matmul) + sum(stats.flops.nonmatmul);

  // ---------- state ----------
  const $ = (id) => document.getElementById(id);
  const canvas = $("canvas"), ctx = canvas.getContext("2d");
  const opts = { showViews: false, showConst: false };
  const view = { x: 0, y: 0, k: 1 };
  const layouts = [];            // layout records, shared by structurally equal modules
  const layoutBySig = new Map(); // structure signature -> index in layouts
  let openSet = new Set([0]);    // open module calls
  let G = null;                  // the graph
  let V = null;                  // what is visible and where, for the current options
  let consumers = null;          // tensor id -> op ids
  let selection = null;          // {kind: "module" | "op" | "tensor", id}
  let busy = false;              // a layout is running
  let C = {};                    // theme colors
  let W = 0, H = 0, dpr = 1;
  let elk = null;

  function readColors() {
    const cs = getComputedStyle(document.documentElement);
    for (const k of ["surface", "surface-alt", "surface-elevated", "fill", "ink", "ink-secondary", "line",
      "line-strong", "brand", "chart-1", "chart-2", "bar-mono-dim", "wire-muted"]) {
      C[k] = cs.getPropertyValue("--" + k).trim();
    }
    // Matmul and non-matmul are two categories to compare: chart-1 and chart-2. Ops without FLOPs are gray.
    C.cat = { matmul: C["chart-1"], nonmatmul: C["chart-2"], none: C["bar-mono-dim"] };
  }
  // Module grounds alternate white and gray with depth, so a box stands out from its container.
  const ground = (mod) => (mod.depth % 2 === 0 ? C.surface : C["surface-alt"]);

  // ---------- derived view model ----------
  // Hidden ops: view ops (unless "Views") and constant-foldable ops (unless "Constants"). An edge
  // through a hidden view op connects the producer of the view input to the consumer of the view.
  function derive() {
    const T = G.tensors, O = G.ops, M = G.modules;
    const hidden = O.map((o) => (o.const && !opts.showConst) || (o.view && !opts.showViews));
    const anc = M.map(() => null);
    const ancestors = (m) => anc[m] || (anc[m] = M[m].parent === null ? [m] : [...ancestors(M[m].parent), m]);
    const resolve = (t) => {
      let p;
      while ((p = T[t].producer) !== null && hidden[p] && O[p].view && O[p].inputs.length) t = O[p].inputs[0];
      return t;
    };
    const inPorts = M.map(() => new Map()), outPorts = M.map(() => new Map());
    const raw = M.map(() => new Map()); // module -> edge key -> [src, srcPort, dst, dstPort, tensor]
    const visibleOps = M.map(() => 0), firstOp = M.map(() => Infinity), directOps = M.map(() => []);
    for (const o of O) {
      if (hidden[o.id]) continue;
      directOps[o.module].push(o.id);
      for (const m of ancestors(o.module)) {
        visibleOps[m]++;
        firstOp[m] = Math.min(firstOp[m], o.id);
      }
    }
    const addRaw = (m, s, sp, d, dp, t) => {
      const key = `${s}.${sp}>${d}.${dp}`;
      if (!raw[m].has(key)) raw[m].set(key, [s, sp, d, dp, t]);
    };
    // Add an edge for tensor t from op p (-1: a graph input) to op c (-1: a graph output), at the
    // lowest common module and through the module boundaries on each side.
    const addEdge = (p, t, c) => {
      const ap = p >= 0 ? ancestors(O[p].module) : [0];
      const ac = c >= 0 ? ancestors(O[c].module) : [0];
      let k = 0;
      while (k < ap.length && k < ac.length && ap[k] === ac[k]) k++;
      const lca = ac[k - 1];
      if (p < 0) inPorts[0].set(t, true);
      if (c < 0) outPorts[0].set(t, true);
      const src = p < 0 ? ["i" + t, -1] : k < ap.length ? ["m" + ap[k], t] : ["o" + p, -1];
      const dst = c < 0 ? ["x" + t, -1] : k < ac.length ? ["m" + ac[k], t] : ["o" + c, -1];
      addRaw(lca, src[0], src[1], dst[0], dst[1], t);
      for (let j = k; j < ap.length; j++) {
        outPorts[ap[j]].set(t, true);
        addRaw(ap[j], j + 1 < ap.length ? "m" + ap[j + 1] : "o" + p, j + 1 < ap.length ? t : -1, "x" + t, -1, t);
      }
      for (let j = k; j < ac.length; j++) {
        inPorts[ac[j]].set(t, true);
        addRaw(ac[j], "i" + t, -1, j + 1 < ac.length ? "m" + ac[j + 1] : "o" + c, j + 1 < ac.length ? t : -1, t);
      }
    };
    // Ports and edges carry the tensor that the producer writes, so the views that different
    // consumers take of one tensor share one port.
    for (const o of O) {
      if (hidden[o.id]) continue;
      for (const r of new Set(o.inputs.map(resolve))) {
        const p = T[r].producer;
        if (p === null) { if (T[r].kind === "input") addEdge(-1, r, o.id); }
        else if (!hidden[p]) addEdge(p, r, o.id);
      }
    }
    for (const r of new Set(G.outputs.map(resolve))) {
      const p = T[r].producer;
      if (p === null) { if (T[r].kind === "input") addEdge(-1, r, -1); }
      else if (!hidden[p]) addEdge(p, r, -1);
    }
    // Items of each module: input ports, then ops and child modules in execution order, then output ports.
    const items = M.map(() => null), edges = M.map(() => null), portIndex = M.map(() => null);
    for (const mod of M) {
      const m = mod.id;
      if (!visibleOps[m] && m !== 0) continue;
      const body = [
        ...directOps[m].map((o) => ({ kind: "op", id: o, order: o })),
        ...mod.children.filter((c) => visibleOps[c] > 0).map((c) => ({ kind: "mod", id: c, order: firstOp[c] })),
      ].sort((a, b) => a.order - b.order);
      const ins = [...inPorts[m].keys()], outs = [...outPorts[m].keys()];
      items[m] = [...ins.map((t) => ({ kind: "in", id: t })), ...body, ...outs.map((t) => ({ kind: "out", id: t }))];
      portIndex[m] = { in: new Map(ins.map((t, i) => [t, i])), out: new Map(outs.map((t, i) => [t, i])) };
    }
    for (const mod of M) {
      const m = mod.id;
      if (!items[m]) continue;
      const index = new Map(items[m].map((it, i) => [(it.kind === "op" ? "o" : it.kind === "mod" ? "m" : it.kind === "in" ? "i" : "x") + it.id, i]));
      edges[m] = [...raw[m].values()].map(([s, sp, d, dp, t]) => ({
        s: index.get(s), sp: sp === -1 ? -1 : portIndex[+s.slice(1)].out.get(sp),
        d: index.get(d), dp: dp === -1 ? -1 : portIndex[+d.slice(1)].in.get(dp), t,
      })).filter((e) => e.s !== undefined && e.d !== undefined)
        .sort((a, b) => a.s - b.s || a.sp - b.sp || a.d - b.d || a.dp - b.dp);
    }
    const siblingMax = M.map(() => 0);
    for (const mod of M) {
      if (!items[mod.id]) continue;
      const kids = items[mod.id].filter((it) => it.kind === "mod").map((it) => it.id);
      const max = Math.max(0, ...kids.map((c) => totalFlops(M[c].stats)));
      for (const c of kids) siblingMax[c] = max;
    }
    return { hidden, resolve, items, edges, portIndex, siblingMax, ancestors, layoutId: M.map(() => -1), absMod: new Map(), absOp: new Map() };
  }

  const childModules = (m) => V.items[m].filter((it) => it.kind === "mod").map((it) => it.id);
  const isOpen = (m) => m === 0 || openSet.has(m);

  // The first view: open module levels, breadth first, while the visible items stay few.
  function startOpenSet() {
    const open = new Set([0]);
    let frontier = [0], count = V.items[0].length;
    for (;;) {
      const next = frontier.flatMap(childModules);
      if (!next.length) break;
      const add = next.reduce((s, c) => s + V.items[c].length - 1, 0);
      if (count + add > START_ITEMS && next.length > 1) break;  // always open a lone wrapper module
      next.forEach((c) => open.add(c));
      count += add;
      frontier = next;
    }
    return open;
  }

  // ---------- layout ----------
  function cardLayout(m) {
    const nIn = V.portIndex[m].in.size, nOut = V.portIndex[m].out.size;
    const sig = `card:${nIn}:${nOut}`;
    if (!layoutBySig.has(sig)) {
      const spread = (n) => Array.from({ length: n }, (_, i) => (CARD_W * (i + 1)) / (n + 1));
      layouts.push({ card: true, w: CARD_W, h: CARD_H, inX: spread(nIn), outX: spread(nOut) });
      layoutBySig.set(sig, layouts.length - 1);
    }
    return layoutBySig.get(sig);
  }

  // The inputs and outputs of a module get narrower pills when there are many of them (the KV cache
  // of all layers that goes into a decode step). The tooltip and the details panel show the full label.
  function portWidth(m, kind) {
    const n = V.portIndex[m][kind].size;
    return n <= 8 ? PORT_W : n <= 16 ? 140 : 96;
  }

  function signature(m, ids) {
    const kinds = V.items[m].map((it) => (it.kind === "mod" ? "m" + ids[it.id] : it.kind[0])).join(",");
    return kinds + "|" + V.edges[m].map((e) => `${e.s}.${e.sp}>${e.d}.${e.dp}`).join(",");
  }

  function elkGraph(m, ids) {
    const children = V.items[m].map((it, i) => {
      const id = "n" + i;
      if (it.kind === "op") return { id, width: OP_W, height: OP_H };
      if (it.kind === "in" || it.kind === "out") {
        return { id, width: portWidth(m, it.kind), height: PORT_H,
          layoutOptions: { "elk.layered.layering.layerConstraint": it.kind === "in" ? "FIRST" : "LAST" } };
      }
      const L = layouts[ids[it.id]];
      return {
        id, width: L.w, height: L.h, layoutOptions: { "elk.portConstraints": "FIXED_POS" },
        ports: [
          ...L.inX.map((x, k) => ({ id: `${id}i${k}`, x: x - 1, y: -2, width: 2, height: 2, layoutOptions: { "elk.port.side": "NORTH" } })),
          ...L.outX.map((x, k) => ({ id: `${id}o${k}`, x: x - 1, y: L.h, width: 2, height: 2, layoutOptions: { "elk.port.side": "SOUTH" } })),
        ],
      };
    });
    // One ELK edge for each pair of connected items. The other edges of the pair (the KV cache
    // tensors from a prefill to a decode step) follow its route: see bundledEdges.
    const seen = new Set();
    const edges = [];
    V.edges[m].forEach((e, j) => {
      if (seen.has(e.s + ">" + e.d)) return;
      seen.add(e.s + ">" + e.d);
      edges.push({ id: "e" + j, sources: [e.sp >= 0 ? `n${e.s}o${e.sp}` : "n" + e.s],
        targets: [e.dp >= 0 ? `n${e.d}i${e.dp}` : "n" + e.d] });
    });
    return { id: "root", layoutOptions: { ...ELK_OPTIONS }, children, edges };
  }

  // Route each edge that ELK did not lay out along the edge of the same pair of items that it did,
  // from its own source port to its own target port.
  function bundledEdges(m, ids, pos, edgePts) {
    const routed = new Map();
    V.edges[m].forEach((e, j) => { if (edgePts[j] && !routed.has(e.s + ">" + e.d)) routed.set(e.s + ">" + e.d, edgePts[j]); });
    const port = (i, k, out) => {
      const p = pos[i], it = V.items[m][i];
      const x = k >= 0 ? p[0] + layouts[ids[it.id]][out ? "outX" : "inX"][k] : p[0] + p[2] / 2;
      return [x, out ? p[1] + p[3] : p[1]];
    };
    V.edges[m].forEach((e, j) => {
      const base = routed.get(e.s + ">" + e.d);
      if (edgePts[j] || !base) return;
      edgePts[j] = [port(e.s, e.sp, true), ...base.slice(1, -1), port(e.d, e.dp, false)];
    });
  }

  function layoutRecord(m, res, ids) {
    const pos = new Array(V.items[m].length);
    for (const c of res.children) pos[+c.id.slice(1)] = [c.x, c.y, c.width, c.height];
    const edgePts = new Array(V.edges[m].length);
    for (const e of res.edges) {
      const pts = [];
      for (const s of e.sections || []) {
        pts.push([s.startPoint.x, s.startPoint.y]);
        for (const b of s.bendPoints || []) pts.push([b.x, b.y]);
        pts.push([s.endPoint.x, s.endPoint.y]);
      }
      edgePts[+e.id.slice(1)] = pts;
    }
    bundledEdges(m, ids, pos, edgePts);
    return finishRecord(m, res.width, res.height, pos, edgePts);
  }

  // If ELK fails, stack the items in one column and draw straight edges.
  function columnLayout(m, ids) {
    let y = PAD_TOP, w = 0;
    const pos = V.items[m].map((it) => {
      const L = it.kind === "mod" ? layouts[ids[it.id]] : null;
      const [iw, ih] = L ? [L.w, L.h] : it.kind === "op" ? [OP_W, OP_H] : [portWidth(m, it.kind), PORT_H];
      const p = [PAD, y, iw, ih];
      y += ih + 30;
      w = Math.max(w, iw);
      return p;
    });
    const edgePts = V.edges[m].map((e) => {
      const a = pos[e.s], b = pos[e.d];
      return [[a[0] + a[2] / 2, a[1] + a[3]], [b[0] + b[2] / 2, b[1]]];
    });
    return finishRecord(m, w + 2 * PAD, y - 30 + PAD, pos, edgePts);
  }

  function finishRecord(m, w, h, pos, edges) {
    const inX = [], outX = [];
    V.items[m].forEach((it, i) => {
      if (it.kind === "in") inX.push(pos[i][0] + pos[i][2] / 2);
      else if (it.kind === "out") outX.push(pos[i][0] + pos[i][2] / 2);
    });
    return { w, h, pos, edges, inX, outX };
  }

  // Lay out the open modules. The new layout replaces the old layout at the end, so a draw during an
  // ELK run shows the old layout.
  async function layoutAll() {
    const ids = G.modules.map(() => -1);
    const order = [];
    const post = (m) => { if (isOpen(m)) for (const c of childModules(m)) post(c); order.push(m); };
    post(0);
    let fresh = 0;
    for (const m of order) {
      if (!isOpen(m)) { ids[m] = cardLayout(m); continue; }
      const sig = signature(m, ids);
      if (!layoutBySig.has(sig)) {
        let rec = null;
        const small = V.items[m].length <= 40 && V.edges[m].length <= 60;
        for (const options of [small ? SMALL_OPTIONS : {}, FALLBACK_OPTIONS]) {
          try {
            const graph = elkGraph(m, ids);
            Object.assign(graph.layoutOptions, options);
            rec = layoutRecord(m, await elk.layout(graph), ids);
            break;
          } catch (err) {
            console.warn("ELK layout failed for module", G.modules[m].path, String(err));
          }
        }
        layouts.push(rec || columnLayout(m, ids));
        layoutBySig.set(sig, layouts.length - 1);
        if (++fresh % 20 === 0) status(`Laying out modules: ${fresh}`);
      }
      ids[m] = layoutBySig.get(sig);
    }
    const absMod = new Map(), absOp = new Map();
    const walk = (m, x, y) => {
      const L = layouts[ids[m]];
      absMod.set(m, [x, y, L.w, L.h]);
      if (L.card) return;
      V.items[m].forEach((it, i) => {
        const p = L.pos[i];
        if (it.kind === "mod") walk(it.id, x + p[0], y + p[1]);
        else if (it.kind === "op") absOp.set(it.id, [x + p[0], y + p[1], p[2], p[3]]);
      });
    };
    walk(0, 0, 0);
    Object.assign(V, { layoutId: ids, absMod, absOp });
  }

  const screenRect = (m) => {
    const [x, y, w, h] = V.absMod.get(m);
    return [x * view.k + view.x, y * view.k + view.y, w * view.k, h * view.k];
  };

  // Open or close module calls and lay out again. The anchor module keeps its top-left corner and
  // its width on screen, so it grows or shrinks in place.
  async function setOpen(ms, open, anchor = null) {
    if (busy || !V || !V.ready) return;
    busy = true;
    const before = anchor !== null && V.absMod.has(anchor) ? screenRect(anchor) : null;
    for (const m of ms) {
      if (m === 0) continue;
      if (open) for (const a of V.ancestors(m)) openSet.add(a);
      else openSet.delete(m);
    }
    try {
      await layoutAll();
    } finally {
      busy = false;
    }
    if (before && V.absMod.has(anchor)) {
      const [x, y, w] = V.absMod.get(anchor);
      const k = Math.max(0.002, Math.min(8, before[2] / w));
      view.k = k;
      view.x = before[0] - x * k;
      view.y = before[1] - y * k;
    }
    status("");
    buildTree();
    requestDraw();
  }

  // ---------- drawing ----------
  let drawPending = false;
  function requestDraw() {
    if (!drawPending) { drawPending = true; requestAnimationFrame(() => { drawPending = false; draw(); }); }
  }

  function resize() {
    const r = canvas.getBoundingClientRect();
    dpr = window.devicePixelRatio || 1;
    W = r.width; H = r.height;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    requestDraw();
  }

  function roundRect(x, y, w, h, r) {
    r = Math.max(0, Math.min(r, w / 2, h / 2));
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, r);
  }

  function fitText(s, maxW) {
    if (maxW <= 8) return "";
    if (ctx.measureText(s).width <= maxW) return s;
    let lo = 0, hi = s.length;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (ctx.measureText(s.slice(0, mid) + "…").width <= maxW) lo = mid; else hi = mid - 1;
    }
    return lo ? s.slice(0, lo) + "…" : "";
  }

  function text(s, x, y, maxW, font, color, align = "left") {
    ctx.font = font;
    const t = fitText(s, maxW);
    if (!t) return 0;
    ctx.fillStyle = color;
    ctx.textAlign = align;
    ctx.fillText(t, x, y);
    return ctx.measureText(t).width;
  }

  // Diagram text has fixed pixel sizes (13, 12, 11) and is not drawn below 11px.
  const font = (px, weight = 400, mono = false) => `${weight} ${px}px ${mono ? MONO_FONT : TEXT_FONT}`;

  function draw() {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = C["surface-alt"];
    ctx.fillRect(0, 0, W, H);
    if (!V || !V.ready) return;
    ctx.textBaseline = "middle";
    drawModule(0, 0, 0);
  }

  const isSel = (kind, id) => selection && selection.kind === kind && selection.id === id;
  const moduleLabels = (mod) => mod.kind === "composite" ? [opShort(mod.class), "decomposed op"]
    : mod.kind === "function" ? [mod.name, "function"]
    : mod.id === 0 ? [mod.name, ""] : [mod.name, classShort(mod.class)];
  // Boxes that are not module calls have a dashed border: a decomposed op long dashes, a function dots.
  const frameDash = (mod) => (mod.kind === "composite" ? [6, 4] : mod.kind === "function" ? [2, 3] : []);

  function drawModule(m, wx, wy) {
    const L = layouts[V.layoutId[m]], k = view.k;
    const sx = wx * k + view.x, sy = wy * k + view.y, sw = L.w * k, sh = L.h * k;
    if (sx > W || sy > H || sx + sw < 0 || sy + sh < 0) return;
    const mod = G.modules[m];
    if (L.card) { drawCard(mod, sx, sy, sw, sh); return; }

    frame(mod, sx, sy, sw, sh, Math.min(12, 12 * k));
    if (sh < 4) return;
    const band = PAD_TOP * k;
    if (band >= 16) {
      const px = band >= 26 ? 13 : 11, y = sy + band * 0.45, inset = Math.max(8, 12 * k);
      const x0 = sx + inset, maxW = sw - 2 * inset;
      const [title, sub] = moduleLabels(mod);
      let used = text(title, x0, y, maxW, font(px, 600), C.ink) + 8;
      used += text(sub, x0 + used, y, maxW - used, font(px), C["ink-secondary"]) + 16;
      text(summary(mod.stats), x0 + used, y, maxW - used, font(11, 400, true), C["ink-secondary"]);
    }

    // Wires, under the items. The wires of the selected op or tensor are the one red path.
    const E = V.edges[m];
    const sel = selectedTensors();
    const highlighted = [];
    ctx.lineWidth = 1;
    ctx.strokeStyle = C["wire-muted"];
    ctx.fillStyle = C["wire-muted"];
    ctx.beginPath();
    L.edges.forEach((pts, j) => {
      if (!pts || pts.length < 2) return;
      if (sel.has(E[j].t)) highlighted.push(pts); else pathEdge(pts, wx, wy);
    });
    ctx.stroke();
    if (k >= 0.2) L.edges.forEach((pts, j) => { if (pts && pts.length >= 2 && !sel.has(E[j].t)) arrow(pts, wx, wy); });
    if (highlighted.length) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = C.brand;
      ctx.fillStyle = C.brand;
      ctx.beginPath();
      for (const pts of highlighted) pathEdge(pts, wx, wy);
      ctx.stroke();
      for (const pts of highlighted) arrow(pts, wx, wy);
    }

    V.items[m].forEach((it, i) => {
      const p = L.pos[i];
      const x = wx + p[0], y = wy + p[1];
      if (it.kind === "mod") return drawModule(it.id, x, y);
      const ix = x * k + view.x, iy = y * k + view.y, iw = p[2] * k, ih = p[3] * k;
      if (ix > W || iy > H || ix + iw < 0 || iy + ih < 0) return;
      if (it.kind === "op") drawOp(G.ops[it.id], ix, iy, iw, ih);
      else drawPort(it, ix, iy, iw, ih, m === 0 ? null : it.kind === "in" ? sy : sy + sh);
    });
  }

  function frame(mod, x, y, w, h, r) {
    roundRect(x, y, w, h, r);
    ctx.fillStyle = ground(mod);
    ctx.fill();
    const sel = isSel("module", mod.id);
    ctx.lineWidth = sel ? 2 : 1;
    ctx.strokeStyle = sel ? C.brand : C.line;
    ctx.setLineDash(frameDash(mod));
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function pathEdge(pts, wx, wy) {
    const k = view.k;
    ctx.moveTo((wx + pts[0][0]) * k + view.x, (wy + pts[0][1]) * k + view.y);
    for (let i = 1; i < pts.length; i++) ctx.lineTo((wx + pts[i][0]) * k + view.x, (wy + pts[i][1]) * k + view.y);
  }

  function arrow(pts, wx, wy) {
    const k = view.k, n = pts.length;
    const x1 = (wx + pts[n - 1][0]) * k + view.x, y1 = (wy + pts[n - 1][1]) * k + view.y;
    const x0 = (wx + pts[n - 2][0]) * k + view.x, y0 = (wy + pts[n - 2][1]) * k + view.y;
    const a = Math.atan2(y1 - y0, x1 - x0), s = Math.min(7, 22 * k);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - s * Math.cos(a - 0.4), y1 - s * Math.sin(a - 0.4));
    ctx.lineTo(x1 - s * Math.cos(a + 0.4), y1 - s * Math.sin(a + 0.4));
    ctx.closePath();
    ctx.fill();
  }

  function summary(stats) {
    const parts = [];
    const f = totalFlops(stats);
    if (f) parts.push(fmtFlops(f));
    if (stats.weight_bytes) parts.push("W " + fmtBytes(stats.weight_bytes));
    return parts.join("  ·  ");
  }

  // A closed module: name, class, FLOPs, weight traffic, and a bar of its FLOPs (matmul,
  // non-matmul) against its largest sibling.
  function drawCard(mod, sx, sy, sw, sh) {
    frame(mod, sx, sy, sw, sh, Math.min(12, sh * 0.25));
    if (sh < 18 || sw < 48) return;
    const s = mod.stats, f = totalFlops(s), mm = sum(s.flops.matmul);
    const lines = [];
    if (f) lines.push(`${fmtFlops(f)}${mm ? `  ·  ${Math.round((100 * mm) / f)}% matmul` : ""}`);
    if (s.weight_bytes) lines.push(`weights ${fmtBytes(s.weight_bytes)} read`);
    if (!lines.length) lines.push(`${s.ops} ops, no FLOPs`);
    const pad = 12, lh = 18, x0 = sx + pad, maxW = sw - 2 * pad - 16;
    const room = Math.floor((sh - 2 * pad - 14) / lh);
    const n = Math.max(0, Math.min(lines.length, room));
    const showBar = f > 0 && V.siblingMax[mod.id] > 0 && room > n;
    let y = n ? sy + pad + 8 : sy + sh / 2;
    const [title, sub] = moduleLabels(mod);
    const used = text(title, x0, y, maxW, font(13, 600), C.ink);
    text(sub, x0 + used + 8, y, maxW - used - 8, font(13), C["ink-secondary"]);
    if (sh >= 30) text("+", sx + sw - pad, n ? y : sy + sh / 2, 16, font(13, 600), C["ink-secondary"], "right");
    for (let i = 0; i < n; i++) {
      y += lh;
      text(lines[i], x0, y, maxW + 16, font(12), C["ink-secondary"]);
    }
    if (showBar) {
      const full = sw - 2 * pad, bw = full * Math.min(1, f / V.siblingMax[mod.id]), by = y + lh - 4;
      roundRect(x0, by, full, 4, 2);
      ctx.fillStyle = C.fill;
      ctx.fill();
      ctx.fillStyle = C.cat.matmul;
      ctx.fillRect(x0, by, (bw * mm) / f, 4);
      ctx.fillStyle = C.cat.nonmatmul;
      ctx.fillRect(x0 + (bw * mm) / f, by, (bw * (f - mm)) / f, 4);
    }
  }

  // An op: a solid block in its category color while it is too small for text, then a card with
  // a category dot, the op name, FLOPs, output shape and dtype, and the weight bytes it reads.
  function drawOp(op, x, y, w, h) {
    const sel = isSel("op", op.id);
    roundRect(x, y, w, h, Math.min(8, h * 0.18));
    const card = h >= 34 && w >= 90;
    ctx.fillStyle = card ? C["surface-elevated"] : C.cat[op.category];
    ctx.fill();
    ctx.lineWidth = sel ? 2 : 1;
    ctx.strokeStyle = sel ? C.brand : card ? C.line : C.cat[op.category];
    ctx.setLineDash(op.view ? [6, 4] : op.const ? [2, 2] : []);
    ctx.stroke();
    ctx.setLineDash([]);
    if (!card) return;
    const pad = 12, x0 = x + pad, maxW = w - 2 * pad, y1 = y + h * 0.34, y2 = y + h * 0.7;
    ctx.fillStyle = C.cat[op.category];
    ctx.beginPath();
    ctx.arc(x0 + 4, y1, 4, 0, 2 * Math.PI);
    ctx.fill();
    const f = op.flops.matmul + op.flops.nonmatmul;
    const fl = f ? si(f).trim() : "";
    ctx.font = font(11, 400, true);
    const flW = fl ? ctx.measureText(fl).width + 8 : 0;
    text(opShort(op.target), x0 + 16, y1, maxW - 16 - flW, font(13, 600), C.ink);
    if (fl) text(fl, x0 + maxW, y1, flW, font(11, 400, true), C["ink-secondary"], "right");
    const out = G.tensors[op.outputs[0]];
    let line2 = `${shapeStr(out.shape)} ${out.dtype}` + (op.outputs.length > 1 ? ` +${op.outputs.length - 1}` : "");
    if (op.weight_bytes) line2 += `  W ${fmtBytes(op.weight_bytes)}`;
    text(line2, x0, y2, maxW, font(11, 400, true), C["ink-secondary"]);
  }

  // A module input or output: a pill joined to the container edge by a wire.
  function drawPort(it, x, y, w, h, frameY) {
    const t = G.tensors[it.id];
    if (frameY !== null) {
      ctx.strokeStyle = C["wire-muted"];
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x + w / 2, frameY);
      ctx.lineTo(x + w / 2, it.kind === "in" ? y : y + h);
      ctx.stroke();
    }
    roundRect(x, y, w, h, h / 2);
    ctx.fillStyle = C["surface-elevated"];
    ctx.fill();
    const sel = isSel("tensor", it.id);
    ctx.lineWidth = sel ? 2 : 1;
    ctx.strokeStyle = sel ? C.brand : C["line-strong"];
    ctx.stroke();
    if (h < 18) return;
    const label = (t.kind === "input" ? t.name + " " : "") + shapeStr(t.shape) + " " + t.dtype;
    text(label, x + w / 2, y + h / 2, w - 16, font(11, 400, true), C.ink, "center");
  }

  // ---------- hit testing ----------
  // Returns the deepest item under a screen point, and the path of modules down to it.
  function hitTest(px, py) {
    if (!V || !V.ready) return null;
    const path = [];
    const hit = hitModule(0, 0, 0, px, py, path);
    return hit && { ...hit, path };
  }

  function hitModule(m, wx, wy, px, py, path) {
    const L = layouts[V.layoutId[m]], k = view.k;
    const sx = wx * k + view.x, sy = wy * k + view.y;
    if (px < sx || py < sy || px > sx + L.w * k || py > sy + L.h * k) return null;
    path.push(m);
    if (L.card) return { kind: "module", id: m };
    const items = V.items[m];
    for (let i = 0; i < items.length; i++) {
      const p = L.pos[i], it = items[i];
      const ix = (wx + p[0]) * k + view.x, iy = (wy + p[1]) * k + view.y;
      if (px < ix || py < iy || px > ix + p[2] * k || py > iy + p[3] * k) continue;
      if (it.kind === "mod") return hitModule(it.id, wx + p[0], wy + p[1], px, py, path);
      return it.kind === "op" ? { kind: "op", id: it.id } : { kind: "tensor", id: it.id };
    }
    let best = 5, hit = null;
    L.edges.forEach((pts, j) => {
      if (!pts) return;
      for (let i = 1; i < pts.length; i++) {
        const d = segDist(px, py, (wx + pts[i - 1][0]) * k + view.x, (wy + pts[i - 1][1]) * k + view.y,
          (wx + pts[i][0]) * k + view.x, (wy + pts[i][1]) * k + view.y);
        if (d < best) { best = d; hit = { kind: "tensor", id: V.edges[m][j].t }; }
      }
    });
    return hit || { kind: "module", id: m };
  }

  function segDist(px, py, x0, y0, x1, y1) {
    const dx = x1 - x0, dy = y1 - y0, l2 = dx * dx + dy * dy;
    const t = l2 ? Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / l2)) : 0;
    return Math.hypot(px - x0 - t * dx, py - y0 - t * dy);
  }

  function selectedTensors() {
    if (!selection) return new Set();
    const ts = selection.kind === "tensor" ? [selection.id]
      : selection.kind === "op" ? [...G.ops[selection.id].inputs, ...G.ops[selection.id].outputs] : [];
    return new Set(ts.map(V.resolve));
  }

  // ---------- view control ----------
  function zoomAt(px, py, f) {
    const k = Math.max(0.002, Math.min(8, view.k * f));
    view.x = px - ((px - view.x) * k) / view.k;
    view.y = py - ((py - view.y) * k) / view.k;
    view.k = k;
    requestDraw();
  }

  // When you zoom in, the card under the cursor opens when it is wide on the screen. When you zoom
  // out, the innermost open module under the cursor closes when it is narrow.
  function autoOpenClose(px, py, zoomIn) {
    if (busy) return;
    const hit = hitTest(px, py);
    if (!hit) return;
    if (zoomIn) {
      const m = hit.path[hit.path.length - 1];
      if (hit.kind === "module" && !isOpen(m) && screenRect(m)[2] >= Math.min(OPEN_PX, 0.45 * W)) setOpen([m], true, m);
    } else {
      const m = [...hit.path].reverse().find((x) => x !== 0 && isOpen(x));
      if (m !== undefined) {
        const r = screenRect(m);
        if (r[2] < CLOSE_PX && r[3] < 0.5 * H) setOpen([m], false, m);
      }
    }
  }

  let anim = null;
  function animateTo(rect, maxK = 2) {
    const [x, y, w, h] = rect, margin = 40;
    const k1 = Math.min((W - 2 * margin) / w, (H - 2 * margin) / h, maxK);
    const c0 = [(W / 2 - view.x) / view.k, (H / 2 - view.y) / view.k], c1 = [x + w / 2, y + h / 2];
    const k0 = view.k, t0 = performance.now(), dur = 360;
    if (anim) cancelAnimationFrame(anim);
    const step = (now) => {
      const u = Math.min(1, (now - t0) / dur), e = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
      const k = Math.exp(Math.log(k0) + (Math.log(k1) - Math.log(k0)) * e);
      const cx = c0[0] + (c1[0] - c0[0]) * e, cy = c0[1] + (c1[1] - c0[1]) * e;
      view.k = k; view.x = W / 2 - cx * k; view.y = H / 2 - cy * k;
      draw();
      anim = u < 1 ? requestAnimationFrame(step) : null;
    };
    anim = requestAnimationFrame(step);
  }

  function fit() { if (V && V.ready) animateTo(V.absMod.get(0), 1.5); }

  // Show a module, op or tensor: open the modules around it, then move the view to it.
  async function focus(sel) {
    if (!sel || !V || !V.ready) return;
    if (sel.kind === "tensor") {
      const p = G.tensors[sel.id].producer;
      if (p !== null && !V.hidden[p]) focus({ kind: "op", id: p });
      return;
    }
    const m = sel.kind === "module" ? sel.id : G.ops[sel.id].module;
    if (sel.kind === "op" && V.hidden[sel.id]) return;
    const need = (sel.kind === "module" ? V.ancestors(m).slice(0, -1) : V.ancestors(m)).filter((a) => !isOpen(a));
    if (need.length) await setOpen(need, true);
    if (sel.kind === "module" && V.absMod.has(m)) animateTo(V.absMod.get(m));
    else if (sel.kind === "op" && V.absOp.has(sel.id)) {
      const [x, y, w, h] = V.absOp.get(sel.id);
      animateTo([x - w, y - 3 * h, 3 * w, 7 * h]);
    }
  }

  // ---------- interaction ----------
  const pointers = new Map();
  let dragStart = null, pinchDist = 0;
  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, [e.offsetX, e.offsetY]);
    if (pointers.size === 1) dragStart = { x: e.offsetX, y: e.offsetY, vx: view.x, vy: view.y, moved: false };
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      pinchDist = Math.hypot(a[0] - b[0], a[1] - b[1]);
    }
  });
  canvas.addEventListener("pointermove", (e) => {
    if (pointers.has(e.pointerId)) pointers.set(e.pointerId, [e.offsetX, e.offsetY]);
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
      if (pinchDist) {
        const cx = (a[0] + b[0]) / 2, cy = (a[1] + b[1]) / 2;
        zoomAt(cx, cy, d / pinchDist);
        autoOpenClose(cx, cy, d > pinchDist);
      }
      pinchDist = d;
      if (dragStart) dragStart.moved = true;
      return;
    }
    if (dragStart) {
      const dx = e.offsetX - dragStart.x, dy = e.offsetY - dragStart.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) { dragStart.moved = true; canvas.classList.add("dragging"); }
      if (dragStart.moved) { view.x = dragStart.vx + dx; view.y = dragStart.vy + dy; requestDraw(); }
      return;
    }
    showTooltip(hitTest(e.offsetX, e.offsetY), e.offsetX, e.offsetY);
  });
  const endPointer = (e) => {
    pointers.delete(e.pointerId);
    if (pointers.size < 2) pinchDist = 0;
    if (pointers.size === 0 && dragStart) {
      if (!dragStart.moved) select(hitTest(e.offsetX, e.offsetY));
      dragStart = null;
      canvas.classList.remove("dragging");
    }
  };
  canvas.addEventListener("pointerup", endPointer);
  canvas.addEventListener("pointercancel", endPointer);
  canvas.addEventListener("pointerleave", () => { $("tooltip").hidden = true; });
  // Double-click: open a card, close an open module, or zoom to an op.
  canvas.addEventListener("dblclick", async (e) => {
    const hit = hitTest(e.offsetX, e.offsetY);
    if (!hit) return;
    if (hit.kind === "module" && hit.id !== 0) {
      const opening = !isOpen(hit.id);
      await setOpen([hit.id], opening, hit.id);
      if (opening && V.absMod.has(hit.id)) animateTo(V.absMod.get(hit.id));
    } else if (hit.kind === "op") focus(hit);
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    $("tooltip").hidden = true;
    const scale = e.deltaMode === 1 ? 16 : 1;
    if (!e.ctrlKey && Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
      view.x -= e.deltaX * scale;
      requestDraw();
      return;
    }
    const f = Math.exp(-e.deltaY * scale * (e.ctrlKey ? 0.01 : 0.0018));
    zoomAt(e.offsetX, e.offsetY, f);
    autoOpenClose(e.offsetX, e.offsetY, f > 1);
  }, { passive: false });
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT") {
      if (e.key === "Escape") e.target.blur();
      return;
    }
    if (e.key === "f" || e.key === "F") fit();
    else if (e.key === "/") { e.preventDefault(); $("search").focus(); }
    else if (e.key === "Escape") select(null);
  });

  function showTooltip(h, x, y) {
    const tip = $("tooltip");
    if (!h || h.kind === "module" && h.id === 0) { tip.hidden = true; return; }
    let html = "";
    if (h.kind === "module") {
      const m = G.modules[h.id];
      const [title, sub] = moduleLabels(m);
      html = `<b>${esc(title)}</b> <span class="muted">${esc(sub)}</span><br><span class="mono">${esc(m.path)}</span><br>` +
        `${esc(summary(m.stats) || "no FLOPs")}<br><span class="muted">Double-click to ${isOpen(h.id) ? "close" : "open"}</span>`;
    } else if (h.kind === "op") {
      const o = G.ops[h.id];
      html = `<b>${esc(opShort(o.target))}</b> <span class="mono muted">${esc(o.dtype)}</span><br>` +
        `matmul ${fmtFlops(o.flops.matmul)} · non-matmul ${fmtFlops(o.flops.nonmatmul)}` +
        (o.weight_bytes ? `<br>weights ${fmtBytes(o.weight_bytes)} read` : "");
    } else {
      const t = G.tensors[h.id];
      html = `<span class="mono">${esc(t.name)}</span><br><span class="mono">${shapeStr(t.shape)} ${t.dtype}</span> · ${fmtBytes(t.bytes)}`;
    }
    tip.innerHTML = html;
    tip.hidden = false;
    const r = tip.getBoundingClientRect();
    tip.style.left = Math.min(x + 14, W - r.width - 8) + "px";
    tip.style.top = Math.min(y + 14, H - r.height - 8) + "px";
  }

  function select(sel) {
    selection = sel && { kind: sel.kind, id: sel.id };
    renderDetails();
    highlightTreeRow();
    requestDraw();
  }

  // ---------- side panels ----------
  function link(kind, id, label) { return `<a data-kind="${kind}" data-id="${id}">${esc(label)}</a>`; }

  // A tensor: its name, then its shape, dtype and size. A constant is named by the parameters,
  // buffers or constants it comes from.
  function tensorItem(t, extra = "") {
    const x = G.tensors[t];
    const folded = x.const && x.producer !== null && !G.ops[x.producer].view;
    const label = x.const && x.source ? x.source + (folded ? ` (folded: ${x.name})` : "") : x.name;
    return `<li><span class="mono">${link("tensor", t, label)}</span>${extra}<div class="sub mono">${shapeStr(x.shape)} ${x.dtype} · ${fmtBytes(x.bytes)}</div></li>`;
  }
  const tensorList = (ts, item = (t) => tensorItem(t)) =>
    `<ul class="tensors">${ts.slice(0, 40).map(item).join("")}</ul>${ts.length > 40 ? `<p class="sub">and ${ts.length - 40} more</p>` : ""}`;

  function flopsTable(stats) {
    const keys = [...new Set([...Object.keys(stats.flops.matmul), ...Object.keys(stats.flops.nonmatmul)])];
    if (!keys.length) return "<p class='hint'>No FLOPs.</p>";
    const f = (k) => (stats.flops.matmul[k] || 0) + (stats.flops.nonmatmul[k] || 0);
    keys.sort((a, b) => f(b) - f(a));
    const rows = keys.map((k) => `<tr><td class="mono">${esc(k)}</td><td class="n">${stats.flops.matmul[k] ? fmtFlops(stats.flops.matmul[k]) : "–"}</td><td class="n">${stats.flops.nonmatmul[k] ? fmtFlops(stats.flops.nonmatmul[k]) : "–"}</td></tr>`);
    return `<table><tr><th>dtype pair</th><th class="n">matmul</th><th class="n">non-matmul</th></tr>${rows.join("")}` +
      `<tr><th>total</th><th class="n">${fmtFlops(sum(stats.flops.matmul))}</th><th class="n">${fmtFlops(sum(stats.flops.nonmatmul))}</th></tr></table>`;
  }

  function descendantOps(m) {
    const ms = new Set();
    const walk = (x) => { ms.add(x); for (const c of G.modules[x].children) walk(c); };
    walk(m);
    return G.ops.filter((o) => ms.has(o.module));
  }

  function renderDetails() {
    const el = $("details");
    const sel = selection || { kind: "module", id: 0 };
    let html = "";
    if (sel.kind === "module") {
      const m = G.modules[sel.id], s = m.stats;
      const [title, sub] = moduleLabels(m);
      html += `<h2>${esc(title)}</h2><div class="sub">${esc(m.kind === "function" ? `function ${m.class} in ${m.file}` : m.class)}${sub === "decomposed op" ? " · decomposed op" : ""}</div>`;
      if (m.path) html += `<div class="sub mono">${esc(m.path)}</div>`;
      if (m.parent !== null) html += `<div class="sub">in ${link("module", m.parent, moduleLabels(G.modules[m.parent])[0])}</div>`;
      html += `<h3>FLOPs</h3>${flopsTable(s)}`;
      if (s.const_flops) html += `<p class="sub">Constant-foldable ops: ${fmtFlops(s.const_flops)}, not in the totals.</p>`;
      html += `<h3>Weights</h3><table><tr><td>read (traffic)</td><td class="n">${fmtBytes(s.weight_bytes)}</td></tr>` +
        `<tr><td>unique (footprint)</td><td class="n">${fmtBytes(s.weight_footprint)}</td></tr>` +
        `<tr><td>ops</td><td class="n">${s.ops}</td></tr></table>`;
      const list = (title, ts) => ts.length ? `<h3>${title}</h3>${tensorList(ts)}` : "";
      html += list("Inputs", m.inputs) + list("Outputs", m.outputs) + list(`Weight tensors (${m.weights.length})`, m.weights);
      if (m.children.length) {
        html += `<h3>Children</h3><table>${m.children.map((c) => {
          const [t, st] = moduleLabels(G.modules[c]);
          return `<tr><td>${link("module", c, t)} <span class="sub">${esc(st)}</span></td><td class="n">${fmtFlops(totalFlops(G.modules[c].stats))}</td></tr>`;
        }).join("")}</table>`;
      }
      const hist = new Map();
      for (const o of descendantOps(sel.id)) {
        if (o.const) continue;
        const h = hist.get(o.target) || { n: 0, f: 0 };
        h.n++; h.f += o.flops.matmul + o.flops.nonmatmul;
        hist.set(o.target, h);
      }
      const rows = [...hist.entries()].sort((a, b) => b[1].f - a[1].f || b[1].n - a[1].n);
      html += `<h3>Op types</h3><table><tr><th>op</th><th class="n">count</th><th class="n">FLOPs</th></tr>${rows.map(([t, h]) => `<tr><td class="mono">${esc(opShort(t))}</td><td class="n">${h.n}</td><td class="n">${h.f ? fmtFlops(h.f) : "–"}</td></tr>`).join("")}</table>`;
    } else if (sel.kind === "op") {
      const o = G.ops[sel.id];
      html += `<h2>${esc(opShort(o.target))}</h2><div class="sub mono">${esc(o.target)} · ${esc(o.name)}</div>`;
      html += `<div class="sub">in ${link("module", o.module, moduleLabels(G.modules[o.module])[0])}</div>`;
      const tags = [o.category === "none" ? "no FLOPs" : o.category, o.view && "view", o.const && "constant-foldable"].filter(Boolean);
      html += `<p>${tags.map((t) => `<span class="tag">${esc(t)}</span>`).join("")}</p>`;
      if (o.source) html += `<p class="sub mono">${esc(o.source)}</p>`;
      html += `<h3>FLOPs</h3><table><tr><td>dtype pair</td><td class="n">${esc(o.dtype)}</td></tr>` +
        `<tr><td>matmul</td><td class="n">${fmtFlops(o.flops.matmul)}</td></tr><tr><td>non-matmul</td><td class="n">${fmtFlops(o.flops.nonmatmul)}</td></tr>` +
        `<tr><td>weights read</td><td class="n">${fmtBytes(o.weight_bytes)}</td></tr></table>`;
      html += `<h3>Inputs</h3>${tensorList(o.inputs.map((t, i) => i), (i) => {
        const t = o.inputs[i];
        const w = o.weight_reads.includes(t) ? `<span class="tag inline">weight</span>` : "";
        return tensorItem(t, ` <span class="sub">${esc(o.input_args[i])}</span>${w}`);
      })}`;
      html += `<h3>Outputs</h3>${tensorList(o.outputs)}`;
      const attrs = Object.entries(o.attrs);
      if (attrs.length) html += `<h3>Attributes</h3><table>${attrs.map(([k, v]) => `<tr><td>${esc(k)}</td><td class="n">${esc(JSON.stringify(v))}</td></tr>`).join("")}</table>`;
    } else {
      const t = G.tensors[sel.id];
      html += `<h2 class="mono">${esc(t.name)}</h2><div class="sub">${esc(t.kind)}${t.const ? " · constant" : ""}</div>`;
      html += `<table><tr><td>shape</td><td class="n">${shapeStr(t.shape)}</td></tr><tr><td>dtype</td><td class="n">${t.dtype}</td></tr><tr><td>size</td><td class="n">${fmtBytes(t.bytes)}</td></tr></table>`;
      if (t.source && t.source !== t.name) html += `<p class="sub mono">from ${esc(t.source)}</p>`;
      if (t.producer !== null) html += `<h3>Producer</h3><p>${link("op", t.producer, opShort(G.ops[t.producer].target) + "  (" + G.ops[t.producer].name + ")")}</p>`;
      const cs = consumers[sel.id] || [];
      if (cs.length) html += `<h3>Consumers</h3><p>${cs.map((c) => link("op", c, opShort(G.ops[c].target) + " (" + G.ops[c].name + ")")).join("<br>")}</p>`;
    }
    el.innerHTML = html;
  }

  $("details").addEventListener("click", (e) => {
    const a = e.target.closest("a[data-kind]");
    if (!a) return;
    const sel = { kind: a.dataset.kind, id: +a.dataset.id };
    select(sel);
    focus(sel);
  });

  // Hierarchy tree. It shows the open modules expanded, as the canvas does.
  function buildTree() {
    const el = $("tree");
    const scroll = el.scrollTop;
    el.innerHTML = "";
    const total = Math.max(1, totalFlops(G.modules[0].stats));
    const add = (m, depth) => {
      const mod = G.modules[m];
      const kids = childModules(m);
      const mm = sum(mod.stats.flops.matmul), nm = sum(mod.stats.flops.nonmatmul);
      const [title, sub] = moduleLabels(mod);
      const row = document.createElement("div");
      row.className = "row";
      row.dataset.id = m;
      row.style.paddingLeft = 4 + depth * 10 + "px";
      row.innerHTML = `<span class="tw">${kids.length && m !== 0 ? (isOpen(m) ? "▾" : "▸") : ""}</span>` +
        `<span class="nm">${esc(title)}</span><span class="cl">${esc(sub)}</span>` +
        `<span class="bar" title="Share of the model FLOPs: matmul, non-matmul"><span style="width:${(100 * mm) / total}%;background:var(--chart-1)"></span><span style="width:${(100 * nm) / total}%;background:var(--chart-2)"></span></span>` +
        `<span class="fl">${si(mm + nm).trim()}</span>`;
      el.appendChild(row);
      if (isOpen(m)) for (const c of kids) add(c, depth + 1);
    };
    add(0, 0);
    el.scrollTop = scroll;
    highlightTreeRow();
  }

  $("tree").addEventListener("click", async (e) => {
    const row = e.target.closest(".row");
    if (!row) return;
    const m = +row.dataset.id;
    if (e.target.classList.contains("tw") && e.target.textContent) {
      const opening = !isOpen(m);
      const anchor = V.absMod.has(m) ? m : null;
      await setOpen([m], opening, anchor);
      return;
    }
    select({ kind: "module", id: m });
    focus({ kind: "module", id: m });
  });

  function highlightTreeRow() {
    for (const r of $("tree").querySelectorAll(".row.sel")) r.classList.remove("sel");
    if (!selection) return;
    const m = selection.kind === "module" ? selection.id : selection.kind === "op" ? G.ops[selection.id].module : null;
    const row = m !== null && $("tree").querySelector(`.row[data-id="${m}"]`);
    if (row) { row.classList.add("sel"); row.scrollIntoView({ block: "nearest" }); }
  }

  function renderBar() {
    const s = G.modules[0].stats;
    $("model-name").textContent = G.model.name;
    $("model-meta").textContent = "Core ATen · torch " + G.model.torch_version;
    document.title = G.model.name + " · Model Viewer";
    const chip = (dot, k, v) => `<span class="chip">${dot ? `<span class="dot" style="background:${dot}"></span>` : ""}<span class="k">${k}</span><b>${v}</b></span>`;
    $("totals").innerHTML = chip("var(--chart-1)", "matmul", fmtFlops(sum(s.flops.matmul))) +
      chip("var(--chart-2)", "non-matmul", fmtFlops(sum(s.flops.nonmatmul))) +
      chip(null, "weights read", fmtBytes(s.weight_bytes)) +
      chip(null, "unique", fmtBytes(s.weight_footprint)) + chip(null, "ops", String(s.ops));
    const sw = (fill, stroke, dash) => `<span class="swatch" style="background:${fill};border-color:${stroke}${dash ? ";border-style:dashed" : ""}"></span>`;
    $("legend").innerHTML = [
      [`<span class="dot" style="background:var(--chart-1)"></span>`, "matmul"],
      [`<span class="dot" style="background:var(--chart-2)"></span>`, "non-matmul"],
      [`<span class="dot" style="background:var(--bar-mono-dim);outline:1px solid var(--line-strong)"></span>`, "no FLOPs"],
      [sw("var(--surface)", "var(--line)"), "module (+: closed)"],
      [sw("transparent", "var(--line-strong)", true), "decomposed op, view"],
      [`<span class="swatch" style="background:transparent;border-color:var(--line-strong);border-style:dotted"></span>`, "function"],
      [`<span class="swatch" style="border-radius:6px;background:var(--surface-elevated)"></span>`, "module input or output"],
    ].map(([m, l]) => `<span>${m}${l}</span>`).join("");
  }

  // ---------- search ----------
  let matches = [], matchIndex = -1, lastQuery = "";
  function runSearch(q) {
    if (!V || !V.ready) return;
    q = q.trim().toLowerCase();
    if (q !== lastQuery) {
      lastQuery = q;
      matchIndex = -1;
      matches = !q ? [] : [
        ...G.modules.filter((m) => V.items[m.id] && (m.path + " " + m.name + " " + m.class).toLowerCase().includes(q)).map((m) => ({ kind: "module", id: m.id })),
        ...G.ops.filter((o) => !V.hidden[o.id] && (o.name + " " + o.target).toLowerCase().includes(q)).map((o) => ({ kind: "op", id: o.id })),
      ];
    }
    if (!matches.length) { status(q ? "No match" : "", 1200); return; }
    matchIndex = (matchIndex + 1) % matches.length;
    const sel = matches[matchIndex];
    select(sel);
    focus(sel);
    status(`${matchIndex + 1} / ${matches.length}`, 1200);
  }
  $("search").addEventListener("keydown", (e) => { if (e.key === "Enter") runSearch(e.target.value); });

  // ---------- loading ----------
  let statusTimer = null;
  function status(msg, ms) {
    $("status").textContent = msg;
    clearTimeout(statusTimer);
    if (ms) statusTimer = setTimeout(() => { $("status").textContent = ""; }, ms);
  }

  async function rebuild(first) {
    const center = !first && V && V.ready ? [(W / 2 - view.x) / view.k, (H / 2 - view.y) / view.k] : null;
    V = derive();
    if (first) openSet = startOpenSet();
    openSet = new Set([...openSet].filter((m) => V.items[m]));
    status("Laying out modules…");
    await new Promise((r) => setTimeout(r));
    busy = true;
    try { await layoutAll(); } finally { busy = false; }
    V.ready = true;
    status("");
    buildTree();
    if (center) {
      view.x = W / 2 - center[0] * view.k; view.y = H / 2 - center[1] * view.k;
    } else {
      // Fit the graph if it stays readable. If not, show the top of the graph at a readable zoom.
      const [, , w, h] = V.absMod.get(0);
      const k = Math.min(1.2, (W - 80) / w, (H - 80) / h);
      view.k = k >= 0.35 ? k : 0.75;
      view.x = (W - w * view.k) / 2;
      view.y = k >= 0.35 ? (H - h * view.k) / 2 : 24;
    }
    requestDraw();
  }

  // Loads and rebuilds run one at a time: each one replaces the graph state that the other uses.
  let queue = Promise.resolve();
  const enqueue = (fn) => (queue = queue.catch(() => {}).then(fn));

  const load = (graph) => enqueue(async () => {
    if (!graph || !String(graph.format || "").startsWith("model_viewer.graph/")) {
      status("This file is not a model_viewer graph.");
      return;
    }
    V = null;
    G = graph;
    consumers = G.tensors.map(() => []);
    for (const o of G.ops) for (const t of new Set(o.inputs)) consumers[t].push(o.id);
    selection = null;
    layouts.length = 0;
    layoutBySig.clear();
    lastQuery = "";
    renderBar();
    renderDetails();
    await rebuild(true);
  });

  async function loadFile(file) {
    try { await load(JSON.parse(await file.text())); } catch (err) { status("Cannot read the file: " + err.message); }
  }

  $("file").addEventListener("change", (e) => { if (e.target.files[0]) loadFile(e.target.files[0]); });
  $("stage").addEventListener("dragover", (e) => e.preventDefault());
  $("stage").addEventListener("drop", (e) => { e.preventDefault(); if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]); });
  $("fit").addEventListener("click", fit);
  $("expand-all").addEventListener("click", async () => {
    if (!V || !V.ready) return;
    await setOpen(G.modules.filter((m) => V.items[m.id]).map((m) => m.id), true);
    fit();
  });
  $("collapse-all").addEventListener("click", async () => {
    if (!V || !V.ready || busy) return;
    openSet = startOpenSet();
    await setOpen([], true);
    fit();
  });
  const toggle = (id, key) => $(id).addEventListener("click", (e) => {
    opts[key] = !opts[key];
    e.currentTarget.setAttribute("aria-pressed", String(opts[key]));
    if (G) enqueue(() => rebuild(false));
  });
  toggle("show-views", "showViews");
  toggle("show-const", "showConst");
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { readColors(); requestDraw(); });
  window.addEventListener("resize", resize);

  readColors();
  resize();
  if (typeof ELK === "undefined") { status("The ELK layout library did not load."); return; }
  elk = new ELK();
  // The page embeds its graphs as gzip-compressed JSON in base64 (see render.py). With more than one,
  // a picker selects the graph, and the page address (#slug) keeps the choice.
  const embedded = [...document.querySelectorAll("script.graph-data")].filter((el) => el.textContent.trim());
  const decoded = new Map();
  async function loadEmbedded(el) {
    status("Reading the graph…");
    await new Promise((r) => setTimeout(r));
    if (!decoded.has(el)) {
      const bytes = Uint8Array.from(atob(el.textContent.trim()), (c) => c.charCodeAt(0));
      const text = await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).text();
      decoded.set(el, JSON.parse(text));
    }
    for (const b of $("examples").querySelectorAll("button")) b.setAttribute("aria-checked", String(b.dataset.slug === el.dataset.slug));
    await load(decoded.get(el));
  }
  function showExample() {
    const el = embedded.find((e) => "#" + e.dataset.slug === location.hash) || embedded[0];
    loadEmbedded(el).catch((err) => status("Cannot read the embedded graph: " + err.message));
  }
  if (embedded.length > 1) {
    $("examples").innerHTML = embedded.map((el) => `<button role="radio" aria-checked="false" data-slug="${esc(el.dataset.slug)}">${esc(el.dataset.label)}</button>`).join("");
    $("examples").hidden = false;
    $("examples").addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (b && "#" + b.dataset.slug !== location.hash) location.hash = b.dataset.slug;
    });
    window.addEventListener("hashchange", showExample);
  }
  if (embedded.length) showExample();
})();
