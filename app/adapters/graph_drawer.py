import json
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

def build_and_draw_graph(data: json, save_path: str = None, figsize: tuple = (12, 8),
                         show: bool = True, node_distance: float = 2.0) -> nx.DiGraph:
    """
    Build a directed graph from `data['paths']` and draw it.
    - Nodes: label = node label
    - Edges: source -> target with relation label
    - Colors: each path gets its own color
    - Paths are visually separated
    Returns: networkx.DiGraph
    """
    if not data.get("paths"):
        print("No paths found in JSON. Nothing to draw.")
        return None

    G = nx.DiGraph()
    path_node_map = defaultdict(set)
    path_edge_map = defaultdict(set)
    node_display = {}

    # collect edges and nodes
    all_edge_labels = []
    for p_idx, path in enumerate(data["paths"]):
        path_id = path.get("path_id", f"path_{p_idx}")
        branches = path.get("final_branches", [])
        for branch in branches:
            history = branch.get("history", [])
            for triple in history:
                src = triple.get("source", {})
                rel = triple.get("relation", {})
                tgt = triple.get("target", {})

                src_qid = src.get("qid", str(src.get("label", "")))
                tgt_qid = tgt.get("qid", str(tgt.get("label", "")))
                src_label = src.get("label", src_qid)
                tgt_label = tgt.get("label", tgt_qid)
                rel_label = rel.get("label", rel.get("pid", ""))

                src_key = src_qid
                tgt_key = tgt_qid

                node_display[src_key] = f"{src_label}"
                node_display[tgt_key] = f"{tgt_label}"

                G.add_node(src_key)
                G.add_node(tgt_key)

                if G.has_edge(src_key, tgt_key):
                    G[src_key][tgt_key].setdefault("rel_labels", []).append(rel_label)
                    path_edge_map[path_id].add((src_key, tgt_key))
                else:
                    G.add_edge(src_key, tgt_key, rel_labels=[rel_label])
                    path_edge_map[path_id].add((src_key, tgt_key))

                path_node_map[path_id].add(src_key)
                path_node_map[path_id].add(tgt_key)

                all_edge_labels.append(rel_label)

    # Drawing setup
    fig, ax = plt.subplots(figsize=figsize)

    # Colors for paths
    colors = plt.cm.get_cmap("tab10")
    path_ids = list(path_node_map.keys())
    path_color_map = {pid: colors(i % 10) for i, pid in enumerate(path_ids)}

    # Node color assignment (first path containing node)
    node_colors = {}
    for pid in path_ids:
        for n in path_node_map[pid]:
            if n not in node_colors:
                node_colors[n] = path_color_map[pid]

    # Position nodes: separate paths visually
    pos = {}
    offset_step = 15  # distance between path clusters
    for idx, pid in enumerate(path_ids):
        nodes_in_path = list(path_node_map[pid])
        subG = G.subgraph(nodes_in_path)
        sub_pos = nx.spring_layout(subG, seed=42, k=node_distance, scale=5)
        x_offset = (idx % 3) * offset_step  # 3 paths per row
        y_offset = (idx // 3) * offset_step
        for n, (x, y) in sub_pos.items():
            pos[n] = (x + x_offset, y + y_offset)
            
    pos = {n: (x, -y) for n, (x, y) in pos.items()}
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                           node_size=1200,
                           node_color=[node_colors.get(n, (0.7, 0.7, 0.7, 1.0)) for n in G.nodes()],
                           linewidths=1,
                           edgecolors="k",
                           ax=ax)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, labels=node_display, font_size=9, ax=ax)

    # Prepare edge labels
    edge_labels = {}
    max_label_len = 0
    for u, v, attrs in G.edges(data=True):
        rels = attrs.get("rel_labels", [])
        label = ", ".join(dict.fromkeys(rels))
        edge_labels[(u, v)] = label
        max_label_len = max(max_label_len, len(label))

    # Set font size for edge labels
    if max_label_len <= 10:
        edge_font_size = 12
    elif max_label_len <= 25:
        edge_font_size = 10
    elif max_label_len <= 45:
        edge_font_size = 9
    else:
        edge_font_size = 8

    # Draw directed edges with FancyArrowPatch
    existing_pairs = set(G.edges())
    for u, v in G.edges():
        xyA = pos[u]
        xyB = pos[v]
        rad = 0.2 if (v, u) in existing_pairs else 0.05
        arrow = FancyArrowPatch(xyA, xyB,
                                connectionstyle=f"arc3,rad={rad}",
                                arrowstyle='-|>',
                                mutation_scale=18,
                                linewidth=1.5,
                                color='gray',
                                shrinkA=18, shrinkB=18,
                                zorder=1)
        ax.add_patch(arrow)

    # Draw edge labels with bbox
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.9)
    nx.draw_networkx_edge_labels(G, pos,
                                edge_labels=edge_labels,
                                font_size=edge_font_size,
                                bbox=bbox_props,
                                ax=ax)

    # Legend
    handles = [mpatches.Patch(color=path_color_map[pid], label=f'path {i+1}')
               for i, pid in enumerate(path_ids)]
    if handles:
        ax.legend(handles=handles, title="paths", bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Graph saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return G
