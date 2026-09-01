import json
from collections import defaultdict, deque
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch


_PATH_COLORS = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#000000",
    "#8EBA42", "#E8474C", "#78C679", "#B15928",
]

_BG_COLOR   = "#F7F8FA"
_NODE_EDGE  = "#2B2D42"
_EDGE_COLOR = "#6B7280"
_LABEL_FG   = "#1A1A2E"


def _sanitize(name: str) -> str:
    for ch in r'?:/<>\\|*"':
        name = name.replace(ch, "")
    return name.strip()


def _wrap(text: str, width: int = 14) -> str:
    words, lines, line = text.split(), [], []
    for w in words:
        if sum(len(x) for x in line) + len(line) + len(w) > width:
            lines.append(" ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)


def _multi_source_depths(G: nx.DiGraph, sources) -> dict[str, int]:
    """Hop-distance from the nearest root, following edge direction (BFS)."""
    depth: dict[str, int] = {}
    q = deque()
    for s in sources:
        if s in G and s not in depth:
            depth[s] = 0
            q.append(s)
    while q:
        u = q.popleft()
        for v in G.successors(u):
            if v not in depth:
                depth[v] = depth[u] + 1
                q.append(v)
    return depth


def _layered_layout(
    G: nx.DiGraph,
    roots: set[str],
    x_gap: float = 3.2,
    y_gap: float = 3.5,
    passes: int = 3,
) -> tuple[dict[str, tuple[float, float]], dict[str, int]]:
    """
    Root(s) on row 0 (top), each hop pushes one row down (y = -depth * y_gap).
    Within a row, x is assigned by alternating barycenter sweeps (align to
    parents, then to children) to reduce edge crossings — no external
    layout engine required.
    """
    if not G.number_of_nodes():
        return {}, {}

    depth = _multi_source_depths(G, [r for r in roots if r in G])
    # anything unreachable from a known root (shouldn't normally happen given
    # how branches are built) still gets drawn, appended after the deepest row
    max_depth = max(depth.values(), default=-1)
    for n in G.nodes():
        if n not in depth:
            max_depth += 1
            depth[n] = max_depth

    rows: dict[int, list[str]] = defaultdict(list)
    for n, d in depth.items():
        rows[d].append(n)
    n_rows = max(rows) + 1

    x_of: dict[str, float] = {}
    for d in range(n_rows):
        row = rows.get(d, [])
        start = -(len(row) - 1) * x_gap / 2.0
        for i, node in enumerate(row):
            x_of[node] = start + i * x_gap

    def _reorder(d: int, neighbor_fn):
        row = rows.get(d, [])
        if not row:
            return
        def _bary(n):
            xs = [x_of[m] for m in neighbor_fn(n) if m in x_of]
            return sum(xs) / len(xs) if xs else x_of[n]
        ordered = sorted(row, key=_bary)
        start = -(len(ordered) - 1) * x_gap / 2.0
        for i, node in enumerate(ordered):
            x_of[node] = start + i * x_gap
        rows[d] = ordered

    for p in range(passes):
        if p % 2 == 0:
            for d in range(1, n_rows):
                _reorder(d, G.predecessors)
        else:
            for d in range(n_rows - 2, -1, -1):
                _reorder(d, G.successors)

    pos = {n: (x_of[n], -depth[n] * y_gap) for n in G.nodes()}
    return pos, depth


def build_and_draw_graph(
    data: dict,
    discarded_paths: list | None = None,
    save_path: str | None = None,
    figsize: tuple | None = None,
    show: bool = False,
    node_gap: float = 3.2,
    layer_gap: float = 3.5,
    show_hop_labels: bool = True,
    title: str | None = None,
    dark_mode: bool = False,
) -> nx.DiGraph:
    """
    Build and draw a top-down layered graph: initial (root) node(s) at the
    top, each hop one row further down.

    discarded_paths, if given, should be the *live* list of PathResult
    objects from PathPipeline.run()["paths"] (not data reloaded from disk).
    Each PathResult's .discarded_branches holds every LLM-proposed candidate
    beam search rejected, at every hop; these draw as dashed, muted leaves
    off the accepted path. Draw-time only — PathResult.to_dict() never
    includes them, so the saved JSON is unaffected.
    """
    bg    = "#1E1E2E" if dark_mode else _BG_COLOR
    fg    = "#CDD6F4" if dark_mode else _LABEL_FG
    e_col = "#89B4FA" if dark_mode else _EDGE_COLOR
    leaf_col = "#6C6F85" if dark_mode else "#B5B5BD"

    G = nx.DiGraph()
    path_node_map: dict[str, set] = defaultdict(set)
    path_edge_map: dict[str, set] = defaultdict(set)
    node_display:  dict[str, str] = {}
    source_nodes:  set[str]       = set()
    accepted_edges: set[tuple]    = set()

    def _walk_history(history, path_id, mark_discarded: bool):
        for i, triple in enumerate(history):
            src, rel, tgt = triple.get("source", {}), triple.get("relation", {}), triple.get("target", {})
            src_key = src.get("qid", str(src.get("label", "")))
            tgt_key = tgt.get("qid", str(tgt.get("label", "")))
            rel_label = rel.get("label", rel.get("pid", ""))

            node_display.setdefault(src_key, src.get("label", src_key))
            node_display.setdefault(tgt_key, tgt.get("label", tgt_key))
            G.add_node(src_key)
            G.add_node(tgt_key)

            if i == 0 and not mark_discarded:
                source_nodes.add(src_key)

            if (src_key, tgt_key) in accepted_edges:
                if not mark_discarded:
                    G[src_key][tgt_key].setdefault("rel_labels", []).append(rel_label)
            elif G.has_edge(src_key, tgt_key):
                G[src_key][tgt_key].setdefault("rel_labels", []).append(rel_label)
            else:
                G.add_edge(src_key, tgt_key, rel_labels=[rel_label], is_discarded=mark_discarded)

            if not mark_discarded:
                accepted_edges.add((src_key, tgt_key))

            path_edge_map[path_id].add((src_key, tgt_key))
            path_node_map[path_id].add(src_key)
            path_node_map[path_id].add(tgt_key)

    for p_idx, path in enumerate(data.get("paths", [])):
        path_id = path.get("path_id", f"path_{p_idx}")
        for branch in path.get("final_branches", []):
            _walk_history(branch.get("history", []), path_id, mark_discarded=False)

    if G.number_of_nodes() == 0:
        print("⚠  No nodes found – check that data['paths'] is populated.")
        return G

    if discarded_paths:
        for pr in discarded_paths:
            path_id = getattr(pr, "path_id", None)
            if path_id is None:
                continue
            for branch in getattr(pr, "discarded_branches", []):
                _walk_history(branch.to_dict()["history"], path_id, mark_discarded=True)

    leaf_only_nodes = set(G.nodes()) - {n for e in accepted_edges for n in e}

    # ── Layout: single top-down layered pass over the WHOLE graph ──────────
    pos, node_depth = _layered_layout(G, roots=source_nodes, x_gap=node_gap, y_gap=layer_gap)

    path_ids = list(path_node_map.keys())
    path_color_map = {pid: _PATH_COLORS[i % len(_PATH_COLORS)] for i, pid in enumerate(path_ids)}

    node_path_count: dict[str, int] = defaultdict(int)
    node_first_path: dict[str, str] = {}
    for pid in path_ids:
        for n in path_node_map[pid]:
            node_path_count[n] += 1
            node_first_path.setdefault(n, pid)

    node_list = list(G.nodes())
    node_colors = [
        leaf_col if n in leaf_only_nodes
        else ("#AAAAAA" if node_path_count[n] > 1 else path_color_map[node_first_path[n]])
        for n in node_list
    ]
    node_sizes  = [1600 if n in source_nodes else (650 if n in leaf_only_nodes else 1100) for n in node_list]
    line_widths = [3.0  if n in source_nodes else (1.0 if n in leaf_only_nodes else 1.5) for n in node_list]

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    width_span  = (max(xs) - min(xs)) if xs else 0
    height_span = (max(ys) - min(ys)) if ys else 0
    if figsize is None:
        figsize = (max(10, width_span * 0.85 + 5), max(6, abs(height_span) * 0.9 + 3))

    fig, ax = plt.subplots(figsize=figsize, facecolor=bg)
    ax.set_facecolor(bg)
    if title:
        ax.set_title(title, fontsize=15, fontweight="bold", color=fg, pad=14, loc="left")

    node_collection = nx.draw_networkx_nodes(
        G, pos, nodelist=node_list, node_size=node_sizes, node_color=node_colors,
        linewidths=line_widths, edgecolors=_NODE_EDGE if not dark_mode else "#CDD6F4", ax=ax,
    )
    node_collection.set_zorder(3)

    wrapped_labels = {n: _wrap(node_display.get(n, n)) for n in G.nodes()}
    label_texts = nx.draw_networkx_labels(
        G, pos, labels=wrapped_labels, font_size=8, font_color=fg, font_weight="bold", ax=ax,
    )
    for txt in label_texts.values():
        txt.set_zorder(4)

    existing_pairs = set(G.edges())
    edge_to_path: dict[tuple, str] = {}
    for pid in path_ids:
        for e in path_edge_map[pid]:
            edge_to_path.setdefault(e, pid)

    for u, v in G.edges():
        discarded_edge = G[u][v].get("is_discarded", False)
        if discarded_edge:
            color, ls, lw, alpha, scale, shrink, z = leaf_col, (0, (4, 3)), 1.1, 0.55, 14, 16, 1
        else:
            color = path_color_map.get(edge_to_path.get((u, v), ""), e_col)
            ls, lw, alpha, scale, shrink, z = "solid", 1.8, 0.85, 22, 22, 2
        rad = 0.15 if (v, u) in existing_pairs else 0.0  # mostly straight, since flow is top-down
        arrow = FancyArrowPatch(
            pos[u], pos[v], connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
            mutation_scale=scale, linewidth=lw, linestyle=ls, color=color,
            shrinkA=shrink, shrinkB=shrink, zorder=z, alpha=alpha,
        )
        ax.add_patch(arrow)

    edge_labels = {}
    for u, v, attrs in G.edges(data=True):
        rels = attrs.get("rel_labels", [])
        edge_labels[(u, v)] = "\n".join(dict.fromkeys(rels))

    max_len = max((len(l) for l in edge_labels.values()), default=0)
    e_font  = 11 if max_len <= 12 else 9 if max_len <= 28 else 7
    edge_label_texts = nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=e_font,
        font_color="#333333" if not dark_mode else "#CDD6F4",
        bbox=dict(boxstyle="round,pad=0.3", fc="white" if not dark_mode else "#313244",
                  ec="#CCCCCC" if not dark_mode else "#585B70", alpha=0.95, lw=0.8),
        ax=ax,
    )
    for txt in edge_label_texts.values():
        txt.set_zorder(5)

    # ── Hop-depth row labels on the left margin (matches "Hop 0 / Hop 1 ..." from your logs)
    if show_hop_labels and pos:
        left_x = min(xs) - node_gap
        max_depth = max(node_depth.values())
        for d in range(max_depth + 1):
            ax.text(left_x, -d * layer_gap, f"Hop {d}", fontsize=9, fontweight="bold",
                     color=fg, ha="right", va="center", alpha=0.7)

    legend_paths = path_ids[:12]
    handles = [mpatches.Patch(color=path_color_map[pid], label=f"Path {i+1}  ({pid})")
               for i, pid in enumerate(legend_paths)]
    if len(path_ids) > 12:
        handles.append(mpatches.Patch(color="grey", label=f"… +{len(path_ids)-12} more"))

    shared_handle = mpatches.Patch(color="#AAAAAA", label="Shared node")
    source_handle = mpatches.Patch(facecolor="white", edgecolor=_NODE_EDGE, linewidth=2,
                                    label="Path source  (larger ring)")
    leaf_handle = mpatches.Patch(color=leaf_col, label="Proposed by LLM, rejected by beam search")

    if handles:
        leg = ax.legend(
            handles=handles + [shared_handle, source_handle, leaf_handle],
            title="Paths", title_fontsize=9, fontsize=8, bbox_to_anchor=(1.02, 1),
            loc="upper left", framealpha=0.9, edgecolor="#CCCCCC",
            facecolor="white" if not dark_mode else "#1E1E2E", labelcolor=fg,
        )
        leg.get_title().set_color(fg)

    fig.text(0.01, 0.005,
              f"Nodes: {G.number_of_nodes()}  │  Edges: {G.number_of_edges()}  │  Paths: {len(path_ids)}",
              fontsize=7, color="#999999", va="bottom")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        IMAGE_DIR = Path("images")
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        clean_path = IMAGE_DIR / _sanitize(save_path)
        plt.savefig(clean_path, dpi=200, bbox_inches="tight", facecolor=bg)
        print(f"✓ Saved → {clean_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return G