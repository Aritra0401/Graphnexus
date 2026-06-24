# GraphNexus 🚀

A lightweight Graph Database and Graph Analytics Platform built using **Python, FastAPI, and Streamlit**.

GraphNexus enables users to create, store, query, visualize, and analyze graph data through a REST API and an interactive web dashboard. The platform includes persistent graph storage and implements a rich collection of graph algorithms commonly used in network analysis, search systems, recommendation engines, social networks, and knowledge graphs.

---

## Features

### Graph Storage

* Persistent graph storage using custom `.gnx` format
* Node and edge creation APIs
* Graph serialization and recovery
* Graph statistics and metadata

### Graph Operations

* Create Nodes
* Create Edges
* Retrieve Nodes
* Retrieve Neighbors
* Graph Traversal

### Graph Algorithms

Implemented algorithms include:

| Category         | Algorithms                                                             |
| ---------------- | ---------------------------------------------------------------------- |
| Traversal        | BFS, DFS                                                               |
| Shortest Path    | Dijkstra, Floyd–Warshall, A* Search                                    |
| Ranking          | PageRank, HITS                                                         |
| Connectivity     | SCC (Strongly Connected Components), WCC (Weakly Connected Components) |
| Graph Ordering   | Topological Sort                                                       |
| Network Analysis | Betweenness Centrality                                                 |
| Optimization     | Minimum Spanning Tree (MST)                                            |

---

## System Architecture

```text
┌──────────────────┐
│  Streamlit UI    │
└────────┬─────────┘
         │ REST API
         ▼
┌──────────────────┐
│     FastAPI      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   GraphNexus     │
│  Storage Engine  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ graphnexus.gnx   │
│ Persistent Store │
└──────────────────┘
```

---

## Tech Stack

### Backend

* Python
* FastAPI
* Pydantic

### Frontend

* Streamlit

### Graph Processing

* NetworkX
* NumPy

### Storage

* Custom `.gnx` persistent graph format

---

## Project Structure

```text
GraphNexus/
│
├── graph_db.py          # Core graph database engine
├── main.py              # FastAPI backend
├── visualizer.py        # Streamlit dashboard
├── test_graphnexus.py   # Unit tests
├── snap_benchmark.py    # Dataset benchmarking
├── graphnexus.gnx       # Persistent graph storage
├── requirements.txt
└── README.md
```

---

## API Endpoints

### Graph Operations

```http
POST /nodes
GET  /nodes
GET  /nodes/{node_id}
POST /edges
GET  /nodes/{node_id}/neighbors
```

### Algorithms

```http
GET /algo/bfs/{start}
GET /algo/dfs/{start}
GET /algo/dijkstra/{src}
GET /algo/pagerank
GET /algo/scc
GET /algo/wcc
GET /algo/topo-sort
GET /algo/betweenness
GET /algo/mst
GET /algo/hits
GET /algo/astar/{src}/{dst}
GET /algo/floyd-warshall
```

### Statistics

```http
GET /stats
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/GraphNexus.git
cd GraphNexus
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Backend

```bash
uvicorn main:app --reload
```

Open Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Running the Frontend

```bash
streamlit run visualizer.py
```

Open:

```text
http://localhost:8501
```

---

## Example Workflow

1. Create nodes using the API or Streamlit dashboard.
2. Add weighted edges between nodes.
3. Run graph algorithms such as Dijkstra or PageRank.
4. Analyze graph statistics.
5. Persist graph data into the `.gnx` storage file.

---

## Applications

* Knowledge Graphs
* Recommendation Systems
* Social Network Analysis
* Fraud Detection Networks
* Transportation Networks
* Citation Networks
* Search & Ranking Systems

---

## Future Improvements

* Cypher-like query language
* Graph indexing optimization
* Distributed graph storage
* Graph embeddings
* Neo4j compatibility
* Real-time graph visualization

---

## Author

**Aritra Biswas**
Chemical Engineering, IIT Kharagpur

Interested in:

* Data Science
* Machine Learning
* Backend Systems
* Graph Analytics
* Quantitative Research
