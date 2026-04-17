"""Network motif analysis — recurring subgraph patterns.

Identifies 3-node motifs (feed-forward, feedback, mutual) and
compares with random networks to find over/under-represented patterns.
"""
import numpy as np
from itertools import permutations
from .loader import SpikeData
from .connectivity import compute_connectivity_graph, connectivity_to_dict

def enumerate_motifs(data: SpikeData, threshold: float = 0.02) -> dict:
    """Enumerate 3-node motifs in the functional connectivity graph."""
    conn = compute_connectivity_graph(data)
    graph = connectivity_to_dict(conn)
    edges = graph.get("edges", [])
    nodes = [n["id"] for n in graph.get("nodes", [])]
    n = len(nodes)

    if n < 3:
        return {"motifs": {}, "reason": "need at least 3 nodes"}

    # Build adjacency
    adj = np.zeros((n, n))
    node_idx = {nid: i for i, nid in enumerate(nodes)}
    for e in edges:
        s, t = node_idx.get(e["source"]), node_idx.get(e["target"])
        if s is not None and t is not None:
            adj[s, t] = 1
            adj[t, s] = 1  # undirected

    # Count 3-node motifs
    motif_counts = {
        "chain": 0,       # A-B-C (no A-C)
        "triangle": 0,    # A-B, B-C, A-C
        "star": 0,        # A-B, A-C (no B-C)
        "disconnected": 0, # only one edge
    }

    total_triads = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                total_triads += 1
                edges_present = int(adj[i,j]) + int(adj[j,k]) + int(adj[i,k])
                if edges_present == 3:
                    motif_counts["triangle"] += 1
                elif edges_present == 2:
                    # Check if it's a chain or star
                    if adj[i,j] and adj[j,k] and not adj[i,k]:
                        motif_counts["chain"] += 1
                    elif adj[i,j] and adj[i,k] and not adj[j,k]:
                        motif_counts["star"] += 1
                    else:
                        motif_counts["chain"] += 1
                elif edges_present == 1:
                    motif_counts["disconnected"] += 1

    # Compare with random (Erdos-Renyi)
    density = float(np.sum(adj) / max(n * (n-1), 1))
    expected_triangles = total_triads * density**3 if total_triads > 0 else 0

    z_triangle = 0.0
    if expected_triangles > 0:
        z_triangle = (motif_counts["triangle"] - expected_triangles) / max(np.sqrt(expected_triangles), 1)

    return {
        "motif_counts": motif_counts,
        "total_triads": total_triads,
        "density": density,
        "triangle_z_score": float(z_triangle),
        "triangle_enrichment": "over-represented" if z_triangle > 2 else "under-represented" if z_triangle < -2 else "as expected",
        "feed_forward_chains": motif_counts["chain"],
        "recurrent_loops": motif_counts["triangle"],
        "network_nodes": n,
        "network_edges": len(edges),
    }
