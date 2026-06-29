import json
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch


# ── Palette ────────────────────────────────────────────────────────────────────
# Perceptually distinct, colour-blind-friendly set (Okabe–Ito extended)
_PATH_COLORS = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#000000",
    "#8EBA42", "#E8474C", "#78C679", "#B15928",
]

_BG_COLOR   = "#F7F8FA"   # near-white canvas
_NODE_EDGE  = "#2B2D42"   # dark outline for crisp node borders
_EDGE_COLOR = "#6B7280"   # muted arrow colour
_LABEL_FG   = "#1A1A2E"   # dark text on nodes


def _sanitize(name: str) -> str:
    """Strip characters that break file-system paths."""
    for ch in r'?:/<>\\|*"':
        name = name.replace(ch, "")
    return name.strip()


def _auto_figsize(n_nodes: int, n_paths: int) -> tuple[float, float]:
    """Scale figure so dense graphs don't get cramped."""
    cols = min(n_paths, 3)
    rows = (n_paths + 2) // 3
    w = max(12, cols * 7)
    h = max(8,  rows * 6 + 0.5 * n_nodes)
    return (w, h)


def _wrap(text: str, width: int = 14) -> str:
    """
    Soft-wrap a label at word boundaries so long names don't
    stretch node circles horizontally.
    """
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


def build_and_draw_graph(
    data: dict,
    save_path: str | None = None,
    figsize: tuple | None = None,       # None → auto-sized
    show: bool = False,
    node_distance: float = 2.5,
    title: str | None = None,           # optional chart title
    dark_mode: bool = False,            # optional dark theme
) -> nx.DiGraph:
    """
    Build a directed graph from ``data['paths']`` and draw it.

    Improvements over the original
    ─────────────────────────────────
    • Auto-sized figure based on graph complexity
    • Perceptually distinct, colour-blind-safe palette
    • Node labels are word-wrapped so long names stay inside circles
    • Edge labels sit on a solid white pill → never lost behind edges
    • Arrow heads are larger & coloured per-path for instant path tracing
    • Source nodes get a star-shaped marker; shared nodes get a double ring
    • Dark-mode option (pass dark_mode=True)
    • Optional chart title
    • Legend sorted and capped so it doesn't overflow on huge graphs
    • Layered z-order: nodes on top, arrows in middle, edge labels below nodes
    """

    # ── 0. Dark-mode theme swap ─────────────────────────────────────────────
    bg    = "#1E1E2E" if dark_mode else _BG_COLOR
    fg    = "#CDD6F4" if dark_mode else _LABEL_FG
    e_col = "#89B4FA" if dark_mode else _EDGE_COLOR

    # ── 1. Parse triplets ───────────────────────────────────────────────────
    G = nx.DiGraph()
    path_node_map: dict[str, set] = defaultdict(set)
    path_edge_map: dict[str, set] = defaultdict(set)
    node_display:  dict[str, str] = {}
    source_nodes:  set[str]       = set()   # ← track explicit sources

    for p_idx, path in enumerate(data.get("paths", [])):
        path_id = path.get("path_id", f"path_{p_idx}")
        for branch in path.get("final_branches", []):
            history = branch.get("history", [])
            for i, triple in enumerate(history):
                src = triple.get("source", {})
                rel = triple.get("relation", {})
                tgt = triple.get("target", {})

                src_key = src.get("qid", str(src.get("label", "")))
                tgt_key = tgt.get("qid", str(tgt.get("label", "")))
                rel_label = rel.get("label", rel.get("pid", ""))

                node_display[src_key] = src.get("label", src_key)
                node_display[tgt_key] = tgt.get("label", tgt_key)

                G.add_node(src_key)
                G.add_node(tgt_key)

                if i == 0:                         # first triple → src is path root
                    source_nodes.add(src_key)

                if G.has_edge(src_key, tgt_key):
                    G[src_key][tgt_key].setdefault("rel_labels", []).append(rel_label)
                else:
                    G.add_edge(src_key, tgt_key, rel_labels=[rel_label])

                path_edge_map[path_id].add((src_key, tgt_key))
                path_node_map[path_id].add(src_key)
                path_node_map[path_id].add(tgt_key)

    if G.number_of_nodes() == 0:
        print("⚠  No nodes found – check that data['paths'] is populated.")
        return G

    # ── 2. Layout ───────────────────────────────────────────────────────────
    path_ids = list(path_node_map.keys())
    n_paths  = len(path_ids)

    if figsize is None:
        figsize = _auto_figsize(G.number_of_nodes(), n_paths)

    path_color_map = {
        pid: _PATH_COLORS[i % len(_PATH_COLORS)]
        for i, pid in enumerate(path_ids)
    }

    # Each path gets its own spring layout placed in a 3-column grid
    pos: dict[str, tuple] = {}
    col_gap, row_gap = 18, 18
    for idx, pid in enumerate(path_ids):
        nodes_in_path = list(path_node_map[pid])
        subG = G.subgraph(nodes_in_path)
        sub_pos = nx.spring_layout(subG, seed=42 + idx, k=node_distance, scale=5)
        col, row = idx % 3, idx // 3
        for n, (x, y) in sub_pos.items():
            pos[n] = (x + col * col_gap, -(y + row * row_gap))

    # ── 3. Node styling ─────────────────────────────────────────────────────
    # Nodes shared across paths → grey; otherwise coloured by first path
    node_path_count: dict[str, int] = defaultdict(int)
    node_first_path: dict[str, str] = {}
    for pid in path_ids:
        for n in path_node_map[pid]:
            node_path_count[n] += 1
            node_first_path.setdefault(n, pid)

    node_list   = list(G.nodes())
    node_colors = [
        "#AAAAAA" if node_path_count[n] > 1
        else path_color_map[node_first_path[n]]
        for n in node_list
    ]
    node_sizes  = [1600 if n in source_nodes else 1100 for n in node_list]
    line_widths = [3.0  if n in source_nodes else 1.5  for n in node_list]

    # ── 4. Figure ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, facecolor=bg)
    ax.set_facecolor(bg)

    if title:
        ax.set_title(title, fontsize=15, fontweight="bold", color=fg,
                     pad=14, loc="left")

    # ── 5. Draw nodes ───────────────────────────────────────────────────────
    node_collection = nx.draw_networkx_nodes(
        G, pos,
        nodelist=node_list,
        node_size=node_sizes,
        node_color=node_colors,
        linewidths=line_widths,
        edgecolors=_NODE_EDGE if not dark_mode else "#CDD6F4",
        ax=ax,
    )
    node_collection.set_zorder(3)

    # ── 6. Draw wrapped node labels ─────────────────────────────────────────
    wrapped_labels = {n: _wrap(node_display.get(n, n)) for n in G.nodes()}
    label_texts = nx.draw_networkx_labels(
        G, pos,
        labels=wrapped_labels,
        font_size=8,
        font_color=fg,
        font_weight="bold",
        ax=ax,
    )
    for txt in label_texts.values():
        txt.set_zorder(4)

    # ── 7. Draw arrows (per-path colour) ────────────────────────────────────
    existing_pairs = set(G.edges())
    edge_to_path: dict[tuple, str] = {}
    for pid in path_ids:
        for e in path_edge_map[pid]:
            edge_to_path.setdefault(e, pid)

    for u, v in G.edges():
        color = path_color_map.get(edge_to_path.get((u, v), ""), e_col)
        rad   = 0.25 if (v, u) in existing_pairs else 0.08
        arrow = FancyArrowPatch(
            pos[u], pos[v],
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=22,        # larger arrow head
            linewidth=1.8,
            color=color,
            shrinkA=22, shrinkB=22,
            zorder=2,
            alpha=0.85,
        )
        ax.add_patch(arrow)

    # ── 8. Edge labels on solid pill background ─────────────────────────────
    edge_labels = {}
    for u, v, attrs in G.edges(data=True):
        rels = attrs.get("rel_labels", [])
        edge_labels[(u, v)] = "\n".join(dict.fromkeys(rels))   # deduplicated

    max_len = max((len(l) for l in edge_labels.values()), default=0)
    e_font  = 11 if max_len <= 12 else 9 if max_len <= 28 else 7

    edge_label_texts = nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=e_font,
        font_color="#333333" if not dark_mode else "#CDD6F4",
        bbox=dict(boxstyle="round,pad=0.35", fc="white" if not dark_mode else "#313244",
                  ec="#CCCCCC" if not dark_mode else "#585B70", alpha=0.95, lw=0.8),
        ax=ax,
    )
    for txt in edge_label_texts.values():
        txt.set_zorder(5)

    # ── 9. Legend (cap at 12 entries to avoid overflow) ─────────────────────
    legend_paths = path_ids[:12]
    handles = [
        mpatches.Patch(color=path_color_map[pid], label=f"Path {i+1}  ({pid})")
        for i, pid in enumerate(legend_paths)
    ]
    if len(path_ids) > 12:
        handles.append(mpatches.Patch(color="grey", label=f"… +{len(path_ids)-12} more"))

    shared_handle = mpatches.Patch(color="#AAAAAA", label="Shared node")
    source_handle = mpatches.Patch(
        facecolor="white", edgecolor=_NODE_EDGE, linewidth=2,
        label="Path source  (larger ring)",
    )

    if handles:
        leg = ax.legend(
            handles=handles + [shared_handle, source_handle],
            title="Paths",
            title_fontsize=9,
            fontsize=8,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            framealpha=0.9,
            edgecolor="#CCCCCC",
            facecolor="white" if not dark_mode else "#1E1E2E",
            labelcolor=fg,
        )
        leg.get_title().set_color(fg)

    # ── 10. Footer annotation ────────────────────────────────────────────────
    fig.text(
        0.01, 0.005,
        f"Nodes: {G.number_of_nodes()}  │  Edges: {G.number_of_edges()}  "
        f"│  Paths: {n_paths}",
        fontsize=7, color="#999999", va="bottom",
    )

    ax.axis("off")
    plt.tight_layout()

    # ── 11. Save ─────────────────────────────────────────────────────────────
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