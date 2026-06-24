"""
GraphNexus FastAPI REST API
Exposes graph operations over HTTP
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from graph_db import GraphNexus

app = FastAPI(
    title="GraphNexus API",
    description="High-performance graph database with 12+ traversal algorithms",
    version="1.0.0"
)

db = GraphNexus("graphnexus.gnx")

# ── Pydantic models ─────────────────────────

class NodeCreate(BaseModel):
    label: str
    properties: Optional[Dict[str, Any]] = {}

class EdgeCreate(BaseModel):
    src: int
    dst: int
    weight: float = 1.0

# ── Node endpoints ──────────────────────────

@app.post("/nodes", status_code=201)
def create_node(body: NodeCreate):
    node_id = db.add_node(body.label, body.properties)
    return {"node_id": node_id, "label": body.label}

@app.get("/nodes/{node_id}")
def get_node(node_id: int):
    rec = db.get_node(node_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Node not found")
    return {"node_id": rec.node_id, "label": rec.label, "properties": rec.properties}

@app.get("/nodes")
def list_nodes():
    return {"nodes": db.all_nodes(), "count": db.node_count()}

# ── Edge endpoints ──────────────────────────

@app.post("/edges", status_code=201)
def create_edge(body: EdgeCreate):
    db.add_edge(body.src, body.dst, body.weight)
    return {"src": body.src, "dst": body.dst, "weight": body.weight}

@app.get("/nodes/{node_id}/neighbors")
def get_neighbors(node_id: int):
    return {"neighbors": db.neighbors(node_id)}

# ── Algorithm endpoints ─────────────────────

@app.get("/algo/bfs/{start}")
def bfs(start: int):
    return {"order": db.bfs(start)}

@app.get("/algo/dfs/{start}")
def dfs(start: int):
    return {"order": db.dfs(start)}

@app.get("/algo/dijkstra/{src}")
def dijkstra(src: int):
    dist = db.dijkstra(src)
    return {"distances": {str(k): (v if v != float('inf') else "inf") for k, v in dist.items()}}

@app.get("/algo/pagerank")
def pagerank(damping: float = 0.85, iterations: int = 100):
    ranks = db.pagerank(damping, iterations)
    return {"pagerank": {str(k): round(v, 6) for k, v in ranks.items()}}

@app.get("/algo/scc")
def scc():
    components = db.scc()
    return {"components": components, "count": len(components)}

@app.get("/algo/wcc")
def wcc():
    components = db.wcc()
    return {"components": components, "count": len(components)}

@app.get("/algo/topo-sort")
def topo_sort():
    order = db.topological_sort()
    if order is None:
        raise HTTPException(status_code=400, detail="Graph contains a cycle")
    return {"topological_order": order}

@app.get("/algo/betweenness")
def betweenness():
    scores = db.betweenness_centrality()
    return {"betweenness": {str(k): round(v, 6) for k, v in scores.items()}}

@app.get("/algo/mst")
def mst():
    edges = db.mst_kruskal()
    return {"mst_edges": [{"u": u, "v": v, "weight": w} for u, v, w in edges]}

@app.get("/algo/hits")
def hits():
    hub, auth = db.hits()
    return {
        "hubs":       {str(k): round(v, 6) for k, v in hub.items()},
        "authorities":{str(k): round(v, 6) for k, v in auth.items()}
    }

@app.get("/algo/astar/{src}/{dst}")
def astar(src: int, dst: int):
    path = db.astar(src, dst)
    if path is None:
        raise HTTPException(status_code=404, detail="No path found")
    return {"path": path}

@app.get("/algo/floyd-warshall")
def floyd_warshall():
    apsp = db.floyd_warshall()
    result = {}
    for (u, v), d in apsp.items():
        result[f"{u}->{v}"] = d if d != float('inf') else "inf"
    return {"apsp": result}

# ── Stats endpoint ──────────────────────────

@app.get("/stats")
def stats():
    return {
        "node_count": db.node_count(),
        "edge_count": db.edge_count(),
        "storage_file": db.storage_path
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
