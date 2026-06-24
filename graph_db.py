"""
GraphNexus - Core Graph Database Engine
Binary-encoded file system with ordered indexing and hash indexing
"""

import struct
import hashlib
import os
import json
import pickle
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple
import heapq
import math


# ─────────────────────────────────────────────
# BINARY FILE SYSTEM & INDEXING
# ─────────────────────────────────────────────

class BinaryNodeRecord:
    """Fixed-size binary record for a node: 256 bytes total."""
    RECORD_SIZE = 256
    HEADER = b'GNX1'  # magic bytes

    def __init__(self, node_id: int, label: str, properties: dict):
        self.node_id = node_id
        self.label = label[:31]          # max 31 chars
        self.properties = properties

    def serialize(self) -> bytes:
        label_bytes = self.label.encode('utf-8').ljust(32, b'\x00')[:32]
        props_json = json.dumps(self.properties).encode('utf-8')
        props_bytes = props_json[:188].ljust(188, b'\x00')   # 188 bytes for props
        header = self.HEADER                                  # 4 bytes
        node_id_bytes = struct.pack('>I', self.node_id)      # 4 bytes
        pad = b'\x00' * (self.RECORD_SIZE - 4 - 4 - 32 - 188)
        return header + node_id_bytes + label_bytes + props_bytes + pad

    @classmethod
    def deserialize(cls, data: bytes) -> 'BinaryNodeRecord':
        assert data[:4] == cls.HEADER, "Invalid record header"
        node_id = struct.unpack('>I', data[4:8])[0]
        label = data[8:40].rstrip(b'\x00').decode('utf-8')
        props_raw = data[40:228].rstrip(b'\x00').decode('utf-8')
        properties = json.loads(props_raw) if props_raw else {}
        return cls(node_id, label, properties)


class OrderedIndex:
    """B-Tree-like ordered index for range queries on node_id."""

    def __init__(self):
        self._keys: List[int] = []          # sorted list of node_ids
        self._offsets: Dict[int, int] = {}  # node_id → byte offset in file

    def insert(self, node_id: int, offset: int):
        lo, hi = 0, len(self._keys)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._keys[mid] < node_id:
                lo = mid + 1
            else:
                hi = mid
        self._keys.insert(lo, node_id)
        self._offsets[node_id] = offset

    def lookup(self, node_id: int) -> Optional[int]:
        lo, hi = 0, len(self._keys)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._keys[mid] == node_id:
                return self._offsets[node_id]
            elif self._keys[mid] < node_id:
                lo = mid + 1
            else:
                hi = mid
        return None

    def range_lookup(self, lo_id: int, hi_id: int) -> List[int]:
        return [k for k in self._keys if lo_id <= k <= hi_id]


class LinearProbeHashIndex:
    """
    Hash index with linear probing for sub-millisecond exact lookups.
    Backed by a flat array of (node_id, offset) slots.
    """
    EMPTY = -1

    def __init__(self, capacity: int = 1024):
        self.capacity = capacity
        self.slots: List[Tuple[int, int]] = [(self.EMPTY, -1)] * capacity
        self.size = 0

    def _hash(self, key: int) -> int:
        return key % self.capacity

    def insert(self, node_id: int, offset: int):
        if self.size / self.capacity > 0.7:
            self._resize()
        idx = self._hash(node_id)
        while self.slots[idx][0] != self.EMPTY:
            if self.slots[idx][0] == node_id:
                self.slots[idx] = (node_id, offset)
                return
            idx = (idx + 1) % self.capacity
        self.slots[idx] = (node_id, offset)
        self.size += 1

    def lookup(self, node_id: int) -> Optional[int]:
        idx = self._hash(node_id)
        probes = 0
        while self.slots[idx][0] != self.EMPTY and probes < self.capacity:
            if self.slots[idx][0] == node_id:
                return self.slots[idx][1]
            idx = (idx + 1) % self.capacity
            probes += 1
        return None

    def _resize(self):
        old_slots = self.slots
        self.capacity *= 2
        self.slots = [(self.EMPTY, -1)] * self.capacity
        self.size = 0
        for node_id, offset in old_slots:
            if node_id != self.EMPTY:
                self.insert(node_id, offset)


# ─────────────────────────────────────────────
# CORE GRAPH DATABASE
# ─────────────────────────────────────────────

class GraphNexus:
    """
    High-performance graph database with:
    - Binary-encoded file system (256-byte fixed records)
    - Ordered indexing (binary search, O(log n))
    - Hash indexing with linear probing (O(1) average)
    - 10+ graph algorithms
    """

    def __init__(self, storage_path: str = "graphnexus.gnx"):
        self.storage_path = storage_path
        self.ordered_index = OrderedIndex()
        self.hash_index = LinearProbeHashIndex(capacity=2048)

        # In-memory adjacency (for algorithm speed)
        self.adj: Dict[int, Dict[int, float]] = defaultdict(dict)   # src→{dst: weight}
        self.radj: Dict[int, Dict[int, float]] = defaultdict(dict)  # dst→{src: weight}
        self.nodes: Dict[int, BinaryNodeRecord] = {}
        self._next_id = 0
        self._file_offset = 0

        if os.path.exists(storage_path):
            self._load_from_disk()

    # ── Storage ──────────────────────────────

    def _write_record(self, record: BinaryNodeRecord) -> int:
        """Write binary record, return byte offset."""
        offset = self._file_offset
        with open(self.storage_path, 'ab') as f:
            f.write(record.serialize())
        self._file_offset += BinaryNodeRecord.RECORD_SIZE
        return offset

    def _read_record(self, offset: int) -> BinaryNodeRecord:
        with open(self.storage_path, 'rb') as f:
            f.seek(offset)
            data = f.read(BinaryNodeRecord.RECORD_SIZE)
        return BinaryNodeRecord.deserialize(data)

    def _load_from_disk(self):
        """Rebuild indices from binary file on startup."""
        if not os.path.exists(self.storage_path):
            return
        with open(self.storage_path, 'rb') as f:
            offset = 0
            while True:
                data = f.read(BinaryNodeRecord.RECORD_SIZE)
                if not data or len(data) < BinaryNodeRecord.RECORD_SIZE:
                    break
                try:
                    rec = BinaryNodeRecord.deserialize(data)
                    self.nodes[rec.node_id] = rec
                    self.ordered_index.insert(rec.node_id, offset)
                    self.hash_index.insert(rec.node_id, offset)
                    self._next_id = max(self._next_id, rec.node_id + 1)
                    offset += BinaryNodeRecord.RECORD_SIZE
                except Exception:
                    break
        self._file_offset = offset

    # ── Node / Edge CRUD ─────────────────────

    def add_node(self, label: str, properties: dict = None) -> int:
        node_id = self._next_id
        self._next_id += 1
        props = properties or {}
        rec = BinaryNodeRecord(node_id, label, props)
        offset = self._write_record(rec)
        self.ordered_index.insert(node_id, offset)
        self.hash_index.insert(node_id, offset)
        self.nodes[node_id] = rec
        return node_id

    def add_edge(self, src: int, dst: int, weight: float = 1.0):
        self.adj[src][dst] = weight
        self.radj[dst][src] = weight

    def get_node(self, node_id: int) -> Optional[BinaryNodeRecord]:
        # Try hash index first (O(1))
        offset = self.hash_index.lookup(node_id)
        if offset is not None:
            return self._read_record(offset)
        return None

    def neighbors(self, node_id: int) -> Dict[int, float]:
        return self.adj.get(node_id, {})

    def all_nodes(self) -> List[int]:
        return list(self.nodes.keys())

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return sum(len(v) for v in self.adj.values())

    # ─────────────────────────────────────────
    # GRAPH ALGORITHMS (10+)
    # ─────────────────────────────────────────

    # 1. BFS
    def bfs(self, start: int) -> List[int]:
        visited, order = set(), []
        q = deque([start])
        visited.add(start)
        while q:
            node = q.popleft()
            order.append(node)
            for nb in self.adj.get(node, {}):
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)
        return order

    # 2. DFS
    def dfs(self, start: int) -> List[int]:
        visited, order = set(), []
        def _dfs(u):
            visited.add(u)
            order.append(u)
            for v in self.adj.get(u, {}):
                if v not in visited:
                    _dfs(v)
        _dfs(start)
        return order

    # 3. Dijkstra SSSP
    def dijkstra(self, src: int) -> Dict[int, float]:
        dist = {n: math.inf for n in self.nodes}
        dist[src] = 0
        pq = [(0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in self.adj.get(u, {}).items():
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    heapq.heappush(pq, (dist[v], v))
        return dist

    # 4. Floyd-Warshall APSP
    def floyd_warshall(self) -> Dict[Tuple[int,int], float]:
        nodes = list(self.nodes.keys())
        n = len(nodes)
        idx = {v: i for i, v in enumerate(nodes)}
        INF = math.inf
        D = [[INF]*n for _ in range(n)]
        for i in range(n):
            D[i][i] = 0
        for u, neighbors in self.adj.items():
            for v, w in neighbors.items():
                D[idx[u]][idx[v]] = w
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if D[i][k] + D[k][j] < D[i][j]:
                        D[i][j] = D[i][k] + D[k][j]
        result = {}
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                result[(u, v)] = D[i][j]
        return result

    # 5. PageRank
    def pagerank(self, damping: float = 0.85, iterations: int = 100,
                 tol: float = 1e-6) -> Dict[int, float]:
        nodes = list(self.nodes.keys())
        n = len(nodes)
        if n == 0:
            return {}
        rank = {v: 1.0 / n for v in nodes}
        for _ in range(iterations):
            new_rank = {}
            for v in nodes:
                incoming = self.radj.get(v, {})
                s = sum(rank[u] / max(len(self.adj[u]), 1) for u in incoming)
                new_rank[v] = (1 - damping) / n + damping * s
            delta = sum(abs(new_rank[v] - rank[v]) for v in nodes)
            rank = new_rank
            if delta < tol:
                break
        return rank

    # 6. SCC (Kosaraju's algorithm)
    def scc(self) -> List[List[int]]:
        visited = set()
        order = []
        def dfs1(u):
            visited.add(u)
            for v in self.adj.get(u, {}):
                if v not in visited:
                    dfs1(v)
            order.append(u)
        for node in self.nodes:
            if node not in visited:
                dfs1(node)
        visited.clear()
        components = []
        def dfs2(u, comp):
            visited.add(u)
            comp.append(u)
            for v in self.radj.get(u, {}):
                if v not in visited:
                    dfs2(v, comp)
        for node in reversed(order):
            if node not in visited:
                comp = []
                dfs2(node, comp)
                components.append(comp)
        return components

    # 7. WCC (Weakly Connected Components) via Union-Find
    def wcc(self) -> List[List[int]]:
        parent = {n: n for n in self.nodes}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            pa, pb = find(a), find(b)
            if pa != pb:
                parent[pa] = pb
        for u, neighbors in self.adj.items():
            for v in neighbors:
                union(u, v)
        groups = defaultdict(list)
        for n in self.nodes:
            groups[find(n)].append(n)
        return list(groups.values())

    # 8. Topological Sort (Kahn's)
    def topological_sort(self) -> Optional[List[int]]:
        in_deg = defaultdict(int)
        for u in self.nodes:
            for v in self.adj.get(u, {}):
                in_deg[v] += 1
        q = deque([n for n in self.nodes if in_deg[n] == 0])
        order = []
        while q:
            u = q.popleft()
            order.append(u)
            for v in self.adj.get(u, {}):
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    q.append(v)
        return order if len(order) == len(self.nodes) else None  # None = cycle

    # 9. Betweenness Centrality (Brandes)
    def betweenness_centrality(self) -> Dict[int, float]:
        nodes = list(self.nodes.keys())
        cb = {v: 0.0 for v in nodes}
        for s in nodes:
            stack, pred = [], {v: [] for v in nodes}
            sigma = {v: 0 for v in nodes}
            sigma[s] = 1
            dist = {v: -1 for v in nodes}
            dist[s] = 0
            q = deque([s])
            while q:
                v = q.popleft()
                stack.append(v)
                for w in self.adj.get(v, {}):
                    if dist[w] < 0:
                        q.append(w)
                        dist[w] = dist[v] + 1
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)
            delta = {v: 0.0 for v in nodes}
            while stack:
                w = stack.pop()
                for v in pred[w]:
                    if sigma[w] > 0:
                        delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                if w != s:
                    cb[w] += delta[w]
        n = len(nodes)
        norm = (n - 1) * (n - 2) if n > 2 else 1
        return {v: cb[v] / norm for v in nodes}

    # 10. Minimum Spanning Tree (Kruskal's)
    def mst_kruskal(self) -> List[Tuple[int, int, float]]:
        edges = [(w, u, v) for u, nbs in self.adj.items() for v, w in nbs.items() if u < v]
        edges.sort()
        parent = {n: n for n in self.nodes}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        mst = []
        for w, u, v in edges:
            pu, pv = find(u), find(v)
            if pu != pv:
                parent[pu] = pv
                mst.append((u, v, w))
        return mst

    # 11. A* Search
    def astar(self, src: int, dst: int,
              heuristic=None) -> Optional[List[int]]:
        if heuristic is None:
            heuristic = lambda u, v: 0
        open_set = [(0, src)]
        came_from = {}
        g = defaultdict(lambda: math.inf)
        g[src] = 0
        while open_set:
            _, u = heapq.heappop(open_set)
            if u == dst:
                path = []
                while u in came_from:
                    path.append(u)
                    u = came_from[u]
                path.append(src)
                return path[::-1]
            for v, w in self.adj.get(u, {}).items():
                tentative = g[u] + w
                if tentative < g[v]:
                    came_from[v] = u
                    g[v] = tentative
                    f = tentative + heuristic(v, dst)
                    heapq.heappush(open_set, (f, v))
        return None

    # 12. HITS (Hubs & Authorities)
    def hits(self, iterations: int = 50) -> Tuple[Dict[int,float], Dict[int,float]]:
        nodes = list(self.nodes.keys())
        hub = {v: 1.0 for v in nodes}
        auth = {v: 1.0 for v in nodes}
        for _ in range(iterations):
            new_auth = {v: sum(hub[u] for u in self.radj.get(v, {})) for v in nodes}
            new_hub  = {v: sum(new_auth[w] for w in self.adj.get(v, {})) for v in nodes}
            norm_a = math.sqrt(sum(x**2 for x in new_auth.values())) or 1
            norm_h = math.sqrt(sum(x**2 for x in new_hub.values())) or 1
            auth = {v: new_auth[v]/norm_a for v in nodes}
            hub  = {v: new_hub[v]/norm_h  for v in nodes}
        return hub, auth
