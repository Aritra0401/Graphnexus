"""
GraphNexus Web Visualizer
Streamlit app — run: streamlit run web/visualizer.py
"""

import streamlit as st
import requests
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

API = "http://localhost:8000"

st.set_page_config(page_title="GraphNexus Visualizer", layout="wide", page_icon="🔗")
st.title("🔗 GraphNexus — Graph Database Visualizer")

# ── Sidebar: build graph ─────────────────────
st.sidebar.header("Build Graph")

with st.sidebar.expander("Add Node"):
    label = st.text_input("Label", "Person")
    props_str = st.text_area("Properties (JSON)", '{"name": "Alice"}')
    if st.button("Add Node"):
        try:
            props = json.loads(props_str)
            r = requests.post(f"{API}/nodes", json={"label": label, "properties": props})
            st.success(f"Node {r.json()['node_id']} created")
        except Exception as e:
            st.error(str(e))

with st.sidebar.expander("Add Edge"):
    src_id = st.number_input("Source ID", min_value=0, step=1)
    dst_id = st.number_input("Dest ID",   min_value=0, step=1)
    weight = st.number_input("Weight", min_value=0.0, value=1.0, step=0.1)
    if st.button("Add Edge"):
        try:
            r = requests.post(f"{API}/edges", json={"src": int(src_id), "dst": int(dst_id), "weight": weight})
            st.success("Edge added")
        except Exception as e:
            st.error(str(e))

with st.sidebar.expander("Load SNAP Dataset"):
    snap_choices = ["Karate Club", "Les Misérables", "Florentine Families"]
    snap_pick = st.selectbox("Dataset", snap_choices)
    if st.button("Load"):
        if snap_pick == "Karate Club":
            G_snap = nx.karate_club_graph()
        elif snap_pick == "Les Misérables":
            G_snap = nx.les_miserables_graph()
        else:
            G_snap = nx.florentine_families_graph()
        for n in G_snap.nodes():
            requests.post(f"{API}/nodes", json={"label": str(n), "properties": {}})
        for u, v in G_snap.edges():
            requests.post(f"{API}/edges", json={"src": u, "dst": v, "weight": 1.0})
        st.success(f"Loaded {len(G_snap.nodes())} nodes, {len(G_snap.edges())} edges")

# ── Main: fetch graph ───────────────────────
try:
    stats_r = requests.get(f"{API}/stats", timeout=2).json()
    st.info(f"Nodes: **{stats_r['node_count']}**  |  Edges: **{stats_r['edge_count']}**")
except Exception:
    st.warning("⚠️ API not running. Start with: `uvicorn api.main:app --reload`")
    st.stop()

nodes_r = requests.get(f"{API}/nodes").json()["nodes"]

# Build local NetworkX graph for visualization
G_vis = nx.DiGraph()
G_vis.add_nodes_from(nodes_r)
for n in nodes_r:
    nb_r = requests.get(f"{API}/nodes/{n}/neighbors").json()["neighbors"]
    for nb, w in nb_r.items():
        G_vis.add_edge(n, int(nb), weight=w)

# ── Tabs ────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Visualization", "🔬 Algorithms", "📋 Node Inspector"])

with tab1:
    st.subheader("Graph Topology")
    layout_choice = st.selectbox("Layout", ["spring", "circular", "kamada_kawai", "spectral"])
    color_by = st.selectbox("Color by", ["degree", "uniform"])

    if len(G_vis.nodes()) > 0:
        fig, ax = plt.subplots(figsize=(10, 7))
        if layout_choice == "spring":
            pos = nx.spring_layout(G_vis, seed=42)
        elif layout_choice == "circular":
            pos = nx.circular_layout(G_vis)
        elif layout_choice == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G_vis)
        else:
            pos = nx.spectral_layout(G_vis)

        if color_by == "degree":
            degrees = dict(G_vis.degree())
            max_d = max(degrees.values()) if degrees else 1
            node_colors = [cm.plasma(degrees.get(n, 0) / max_d) for n in G_vis.nodes()]
        else:
            node_colors = "#4A90D9"

        nx.draw_networkx(
            G_vis, pos, ax=ax,
            node_color=node_colors,
            node_size=600,
            font_size=8,
            edge_color="#AAAAAA",
            arrows=True,
            arrowsize=15,
            width=1.5
        )
        ax.set_facecolor("#0E1117")
        fig.patch.set_facecolor("#0E1117")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info("Add nodes and edges to visualize the graph.")

with tab2:
    st.subheader("Run Algorithms")
    algo = st.selectbox("Algorithm", [
        "PageRank", "BFS", "DFS", "Dijkstra SSSP",
        "SCC", "WCC", "Topological Sort",
        "Betweenness Centrality", "MST (Kruskal)",
        "HITS", "A* Search"
    ])

    if algo in ["BFS", "DFS", "Dijkstra SSSP"]:
        start_node = st.number_input("Start node", min_value=0, step=1)

    if algo == "A* Search":
        a_src = st.number_input("Source", min_value=0, step=1)
        a_dst = st.number_input("Destination", min_value=0, step=1)

    if st.button("▶ Run"):
        with st.spinner("Running..."):
            try:
                if algo == "PageRank":
                    r = requests.get(f"{API}/algo/pagerank").json()
                    pr = r["pagerank"]
                    sorted_pr = sorted(pr.items(), key=lambda x: -float(x[1]))[:20]
                    st.bar_chart({k: float(v) for k, v in sorted_pr})
                    st.json(dict(sorted_pr[:10]))

                elif algo == "BFS":
                    r = requests.get(f"{API}/algo/bfs/{int(start_node)}").json()
                    st.write("BFS order:", r["order"])

                elif algo == "DFS":
                    r = requests.get(f"{API}/algo/dfs/{int(start_node)}").json()
                    st.write("DFS order:", r["order"])

                elif algo == "Dijkstra SSSP":
                    r = requests.get(f"{API}/algo/dijkstra/{int(start_node)}").json()
                    st.json(r["distances"])

                elif algo == "SCC":
                    r = requests.get(f"{API}/algo/scc").json()
                    st.write(f"Found **{r['count']}** strongly connected components")
                    st.json(r["components"][:10])

                elif algo == "WCC":
                    r = requests.get(f"{API}/algo/wcc").json()
                    st.write(f"Found **{r['count']}** weakly connected components")
                    st.json(r["components"][:10])

                elif algo == "Topological Sort":
                    r = requests.get(f"{API}/algo/topo-sort")
                    if r.status_code == 400:
                        st.error("Graph contains a cycle — topological sort not possible.")
                    else:
                        st.write("Topological order:", r.json()["topological_order"])

                elif algo == "Betweenness Centrality":
                    r = requests.get(f"{API}/algo/betweenness").json()
                    bc = r["betweenness"]
                    sorted_bc = sorted(bc.items(), key=lambda x: -float(x[1]))[:20]
                    st.bar_chart({k: float(v) for k, v in sorted_bc})

                elif algo == "MST (Kruskal)":
                    r = requests.get(f"{API}/algo/mst").json()
                    st.write(f"MST has {len(r['mst_edges'])} edges")
                    st.json(r["mst_edges"][:20])

                elif algo == "HITS":
                    r = requests.get(f"{API}/algo/hits").json()
                    top_hubs = sorted(r["hubs"].items(), key=lambda x: -float(x[1]))[:10]
                    top_auth = sorted(r["authorities"].items(), key=lambda x: -float(x[1]))[:10]
                    col1, col2 = st.columns(2)
                    col1.subheader("Top Hubs")
                    col1.json(dict(top_hubs))
                    col2.subheader("Top Authorities")
                    col2.json(dict(top_auth))

                elif algo == "A* Search":
                    r = requests.get(f"{API}/algo/astar/{int(a_src)}/{int(a_dst)}")
                    if r.status_code == 404:
                        st.error("No path found.")
                    else:
                        st.write("Path:", r.json()["path"])

            except Exception as e:
                st.error(f"Error: {e}")

with tab3:
    st.subheader("Node Inspector")
    inspect_id = st.number_input("Node ID", min_value=0, step=1)
    if st.button("Inspect"):
        try:
            r = requests.get(f"{API}/nodes/{int(inspect_id)}").json()
            st.json(r)
            nb = requests.get(f"{API}/nodes/{int(inspect_id)}/neighbors").json()
            st.write("Neighbors:", nb["neighbors"])
        except Exception as e:
            st.error(str(e))
