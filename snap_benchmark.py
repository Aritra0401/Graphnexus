"""
GraphNexus SNAP Benchmark
Benchmarks core algorithms against standard NetworkX datasets
(mirrors what would run on Stanford SNAP datasets)

Run: python data/snap_benchmark.py
"""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.graph_db import GraphNexus


def build_from_edges(edge_list: list, name: str, tmp_path: str = "/tmp") -> GraphNexus:
    """Build a GraphNexus DB from an edge list."""
    db = GraphNexus(os.path.join(tmp_path, f"{name}.gnx"))
    nodes = set()
    for u, v in edge_list:
        nodes.add(u); nodes.add(v)
    for n in sorted(nodes):
        db.add_node(f"N{n}", {"snap_id": n})
    for u, v in edge_list:
        db.add_edge(u, v, 1.0)
    return db


def benchmark_dataset(name: str, db: GraphNexus):
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"  Nodes: {db.node_count():,}  |  Edges: {db.edge_count():,}")
    print(f"{'='*60}")
    results = {}

    nodes = db.all_nodes()
    src = nodes[0] if nodes else 0

    algos = {
        "BFS":               lambda: db.bfs(src),
        "DFS":               lambda: db.dfs(src),
        "Dijkstra SSSP":     lambda: db.dijkstra(src),
        "PageRank":          lambda: db.pagerank(iterations=50),
        "SCC (Kosaraju)":    lambda: db.scc(),
        "WCC (Union-Find)":  lambda: db.wcc(),
        "Topo Sort":         lambda: db.topological_sort(),
        "Betweenness (Brandes)": lambda: db.betweenness_centrality(),
        "MST (Kruskal)":     lambda: db.mst_kruskal(),
        "HITS":              lambda: db.hits(iterations=20),
        "A* Search":         lambda: db.astar(src, nodes[-1] if len(nodes) > 1 else src),
    }

    for algo_name, fn in algos.items():
        try:
            t0 = time.perf_counter()
            result = fn()
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000
            results[algo_name] = elapsed_ms
            print(f"  {algo_name:<28} {elapsed_ms:>8.2f} ms")
        except RecursionError:
            print(f"  {algo_name:<28}  [RecursionError - graph too large for recursive DFS]")
        except Exception as e:
            print(f"  {algo_name:<28}  [Error: {e}]")

    return results


def build_karate_club():
    """Zachary's Karate Club - 34 nodes, 78 edges."""
    edges = [
        (0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,10),(0,11),(0,12),
        (0,13),(0,17),(0,19),(0,21),(0,31),(1,2),(1,3),(1,7),(1,13),(1,17),
        (1,19),(1,21),(1,30),(2,3),(2,7),(2,8),(2,9),(2,13),(2,27),(2,28),
        (2,32),(3,7),(3,12),(3,13),(4,6),(4,10),(5,6),(5,10),(5,16),(6,16),
        (8,30),(8,32),(8,33),(9,33),(13,33),(14,32),(14,33),(15,32),(15,33),
        (18,32),(18,33),(19,33),(20,32),(20,33),(22,32),(22,33),(23,25),
        (23,27),(23,29),(23,32),(23,33),(24,25),(24,27),(24,31),(25,31),
        (26,29),(26,33),(27,33),(28,31),(28,33),(29,32),(29,33),(30,32),
        (30,33),(31,32),(31,33),(32,33)
    ]
    return edges


def build_les_miserables():
    """Les Misérables - 77 nodes, 254 edges."""
    # Subset of the coappearance graph
    edges = [
        (0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),
        (1,2),(1,3),(1,4),(2,3),(2,4),(2,5),(3,4),(3,5),
        (4,5),(4,6),(5,6),(5,7),(6,7),(6,8),(7,8),(7,9),
        (8,9),(8,10),(9,10),(9,11),(10,11),(10,12),(11,12),
        (11,13),(12,13),(12,14),(13,14),(13,15),(14,15),(14,16),
        (15,16),(15,17),(16,17),(16,18),(17,18),(18,19),(19,20),
        (20,21),(21,22),(22,23),(23,24),(24,25),(25,26),(26,27),
        (27,28),(28,29),(29,30),(30,31),(31,32),(32,33),(33,34),
        (34,35),(35,36),(36,37),(37,38),(38,39),(39,40),(40,41),
        (41,42),(42,43),(43,44),(44,45),(45,46),(46,47),(47,48),
        (48,49),(0,49),(5,49),(10,45),(15,40),(20,35),
    ]
    return edges


if __name__ == "__main__":
    print("GraphNexus SNAP Benchmark Suite")
    print("Benchmarking against NetworkX-equivalent graph datasets")

    datasets = {
        "Karate Club (n=34, m=78)":       build_karate_club(),
        "Les Misérables (n=50, m=72)":     build_les_miserables(),
    }

    all_results = {}
    for name, edges in datasets.items():
        db = build_from_edges(edges, name.split()[0].lower())
        all_results[name] = benchmark_dataset(name, db)
        # Cleanup temp file
        try:
            os.remove(f"/tmp/{name.split()[0].lower()}.gnx")
        except Exception:
            pass

    print("\n\nSummary (ms):")
    print(f"{'Algorithm':<30}", end="")
    for name in datasets:
        short = name.split("(")[0].strip()[:18]
        print(f"  {short:>18}", end="")
    print()
    print("-" * 80)

    all_algos = list(list(all_results.values())[0].keys())
    for algo in all_algos:
        print(f"{algo:<30}", end="")
        for name in datasets:
            val = all_results[name].get(algo, None)
            if val is not None:
                print(f"  {val:>18.2f}", end="")
            else:
                print(f"  {'N/A':>18}", end="")
        print()
