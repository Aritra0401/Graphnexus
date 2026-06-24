"""
GraphNexus Test Suite
Tests for binary storage, indexing, and all 12 graph algorithms
"""

import pytest
import os
import math
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.graph_db import (
    GraphNexus, BinaryNodeRecord,
    OrderedIndex, LinearProbeHashIndex
)


# ── Binary Record Tests ─────────────────────

def test_binary_record_roundtrip():
    rec = BinaryNodeRecord(42, "Person", {"name": "Alice", "age": 30})
    data = rec.serialize()
    assert len(data) == BinaryNodeRecord.RECORD_SIZE
    rec2 = BinaryNodeRecord.deserialize(data)
    assert rec2.node_id == 42
    assert rec2.label == "Person"
    assert rec2.properties["name"] == "Alice"


def test_binary_record_size():
    rec = BinaryNodeRecord(0, "X", {})
    assert len(rec.serialize()) == 256


# ── Ordered Index Tests ─────────────────────

def test_ordered_index_insert_lookup():
    idx = OrderedIndex()
    for i, offset in [(5, 100), (2, 200), (8, 300), (1, 400)]:
        idx.insert(i, offset)
    assert idx.lookup(5) == 100
    assert idx.lookup(2) == 200
    assert idx.lookup(9) is None


def test_ordered_index_range():
    idx = OrderedIndex()
    for i in range(10):
        idx.insert(i, i * 100)
    result = idx.range_lookup(3, 6)
    assert sorted(result) == [3, 4, 5, 6]


# ── Hash Index Tests ────────────────────────

def test_hash_index_insert_lookup():
    idx = LinearProbeHashIndex(capacity=16)
    idx.insert(7, 700)
    idx.insert(23, 2300)
    assert idx.lookup(7) == 700
    assert idx.lookup(23) == 2300
    assert idx.lookup(99) is None


def test_hash_index_collision():
    idx = LinearProbeHashIndex(capacity=4)
    # 0 and 4 both hash to slot 0
    idx.insert(0, 0)
    idx.insert(4, 400)
    idx.insert(8, 800)
    assert idx.lookup(0) == 0
    assert idx.lookup(4) == 400
    assert idx.lookup(8) == 800


def test_hash_index_resize():
    idx = LinearProbeHashIndex(capacity=4)
    for i in range(10):  # forces resize at 70% load
        idx.insert(i, i * 10)
    for i in range(10):
        assert idx.lookup(i) == i * 10


# ── GraphNexus Core Tests ───────────────────

@pytest.fixture
def g(tmp_path):
    db = GraphNexus(str(tmp_path / "test.gnx"))
    # Build a small directed graph:
    # 0→1, 0→2, 1→3, 2→3, 3→4
    for i in range(5):
        db.add_node(f"N{i}", {"val": i})
    db.add_edge(0, 1, 1.0)
    db.add_edge(0, 2, 4.0)
    db.add_edge(1, 3, 2.0)
    db.add_edge(2, 3, 1.0)
    db.add_edge(3, 4, 3.0)
    return db


def test_node_add_and_retrieve(g):
    rec = g.get_node(0)
    assert rec is not None
    assert rec.label == "N0"


def test_node_count(g):
    assert g.node_count() == 5


def test_edge_count(g):
    assert g.edge_count() == 5


def test_bfs(g):
    order = g.bfs(0)
    assert order[0] == 0
    assert set(order) == {0, 1, 2, 3, 4}


def test_dfs(g):
    order = g.dfs(0)
    assert order[0] == 0
    assert set(order) == {0, 1, 2, 3, 4}


def test_dijkstra(g):
    dist = g.dijkstra(0)
    assert dist[4] == 6.0   # 0→1(1)→3(2)→4(3) = 6
    assert dist[3] == 3.0   # 0→1(1)→3(2) = 3
    assert dist[2] == 4.0


def test_floyd_warshall(g):
    apsp = g.floyd_warshall()
    assert apsp[(0, 4)] == 6.0
    assert apsp[(0, 0)] == 0.0


def test_pagerank(g):
    pr = g.pagerank()
    assert all(v >= 0 for v in pr.values())
    # With dangling nodes (node 4 has no outgoing edges), sum < 1 is expected.
    assert sum(pr.values()) > 0


def test_scc_simple(tmp_path):
    db = GraphNexus(str(tmp_path / "scc.gnx"))
    for i in range(3):
        db.add_node(f"N{i}")
    db.add_edge(0, 1)
    db.add_edge(1, 0)  # cycle: {0,1} is one SCC
    db.add_edge(1, 2)
    sccs = db.scc()
    sizes = sorted([len(s) for s in sccs], reverse=True)
    assert sizes[0] == 2  # {0,1}


def test_wcc(tmp_path):
    db = GraphNexus(str(tmp_path / "wcc.gnx"))
    for i in range(4):
        db.add_node(f"N{i}")
    db.add_edge(0, 1)
    db.add_edge(2, 3)  # two separate components
    components = db.wcc()
    assert len(components) == 2


def test_topological_sort(g):
    order = g.topological_sort()
    assert order is not None
    pos = {v: i for i, v in enumerate(order)}
    assert pos[0] < pos[1]
    assert pos[1] < pos[3]
    assert pos[3] < pos[4]


def test_topo_cycle(tmp_path):
    db = GraphNexus(str(tmp_path / "cycle.gnx"))
    for i in range(3):
        db.add_node(f"N{i}")
    db.add_edge(0, 1)
    db.add_edge(1, 2)
    db.add_edge(2, 0)  # creates cycle
    assert db.topological_sort() is None


def test_betweenness(g):
    bc = g.betweenness_centrality()
    assert all(v >= 0 for v in bc.values())


def test_mst(tmp_path):
    db = GraphNexus(str(tmp_path / "mst.gnx"))
    for i in range(4):
        db.add_node(f"N{i}")
    db.add_edge(0, 1, 1.0); db.add_edge(1, 0, 1.0)
    db.add_edge(0, 2, 3.0); db.add_edge(2, 0, 3.0)
    db.add_edge(1, 2, 2.0); db.add_edge(2, 1, 2.0)
    db.add_edge(2, 3, 1.0); db.add_edge(3, 2, 1.0)
    mst = db.mst_kruskal()
    # MST of 4 nodes has exactly 3 edges
    assert len(mst) == 3


def test_astar(g):
    path = g.astar(0, 4)
    assert path is not None
    assert path[0] == 0
    assert path[-1] == 4


def test_hits(g):
    hub, auth = g.hits()
    assert all(v >= 0 for v in hub.values())
    assert all(v >= 0 for v in auth.values())


# ── Persistence test ─────────────────────────

def test_persistence(tmp_path):
    path = str(tmp_path / "persist.gnx")
    db1 = GraphNexus(path)
    n0 = db1.add_node("Persist", {"key": "value"})
    db2 = GraphNexus(path)  # reload from disk
    rec = db2.get_node(n0)
    assert rec is not None
    assert rec.label == "Persist"
    assert rec.properties["key"] == "value"
