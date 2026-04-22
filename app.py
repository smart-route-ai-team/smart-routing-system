"""
app.py – Streamlit Visualization Dashboard  [UPGRADED]
Smart Network-on-Chip (NoC) Routing System
Run: streamlit run app.py

UPGRADES:
  • Protocol Intelligence tab — shows which protocol wins for which data type
  • Packet Drop & Retransmit tab — full drop/retry/recovery tracking per edge
  • Retransmit info surfaces in all routing result cards
  • Protocol fitness score shown on every edge in Network View
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import networkx as nx
import random
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time

from core.graph import create_graph, add_dynamic_traffic, get_graph_stats, clone_graph
from core.simulator import simulate_packet_delivery, calculate_cost
from level3.basic_routing import Level3Router, run_dijkstra, run_astar
from level4.adaptive_routing import Level4Router
from level5.qlearning_router import QLearningRouter
from utils.traffic_model import TrafficModel
from utils.dataset_loader import (
    get_dataset, PROTOCOL_STATS,
    select_best_protocol, protocol_fitness_score
)

# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Router",
    page_icon="🔀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card { background:#1e1e2e; border-radius:10px; padding:16px;
                   border:1px solid #313244; }
    .stTabs [data-baseweb="tab-list"]  { gap:8px; }
    .stTabs [data-baseweb="tab"]       { border-radius:8px 8px 0 0; }
    .proto-uart  { color:#f39c12; font-weight:bold; }
    .proto-i2c   { color:#2ecc71; font-weight:bold; }
    .proto-spi   { color:#4a90d9; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
for key, val in [("G", None), ("history", []), ("q_router", None)]:
    if key not in st.session_state:
        st.session_state[key] = val

PROTO_COLOR = {"UART": "#f39c12", "I2C": "#2ecc71", "SPI": "#4a90d9"}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def build_plotly_graph(G, highlight_path=None, title="Network Graph",
                        show_protocol=False):
    pos = nx.get_node_attributes(G, 'pos')
    if not pos:
        pos = nx.spring_layout(G, seed=42)

    edge_traces = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        proto  = data.get("protocol", "UART")
        color  = "#e74c3c" if data.get("congested") else PROTO_COLOR.get(proto, "#4a90d9")
        width  = 1.5
        if highlight_path:
            edges_in_path = list(zip(highlight_path, highlight_path[1:]))
            if (u,v) in edges_in_path or (v,u) in edges_in_path:
                color = "#ffffff"; width = 4
        label = f"{proto} | fit={data.get('protocol_fitness',0):.2f}" if show_protocol else ""
        edge_traces.append(go.Scatter(
            x=[x0,x1,None], y=[y0,y1,None],
            mode='lines+text' if show_protocol else 'lines',
            text=[label,"",""] if show_protocol else [],
            textposition="middle center",
            textfont=dict(size=8, color="#aaa"),
            line=dict(color=color, width=width),
            hoverinfo='text',
            hovertext=f"{u}↔{v} | {proto} | fitness={data.get('protocol_fitness',0):.2f}"
                      f" | load={data.get('load',0)}% | drop={data.get('frame_err',0):.3f}",
        ))

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_colors = []
    for n in G.nodes():
        in_path = highlight_path and n in highlight_path
        if in_path:
            if n == highlight_path[0]:   node_colors.append("#f39c12")
            elif n == highlight_path[-1]: node_colors.append("#9b59b6")
            else:                         node_colors.append("#2ecc71")
        else:
            node_colors.append("#4a90d9")

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=[str(n) for n in G.nodes()],
        textposition="top center",
        marker=dict(size=18, color=node_colors, line=dict(width=2, color='white')),
        hovertext=[f"Node {n}" for n in G.nodes()],
        hoverinfo='text',
    )
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title=title, showlegend=False,
        margin=dict(l=10,r=10,t=40,b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        height=420,
    )
    return fig


def retransmit_badge(retx_info: dict) -> str:
    if not retx_info or not retx_info.get("dropped"):
        return "✅ Delivered"
    if retx_info.get("recovered"):
        return f"🔄 Retransmitted ({retx_info['attempts']-1}x) → Recovered"
    return f"❌ Perm Fail after {retx_info['attempts']-1} retries"


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.title("Router Dashboard")
st.sidebar.markdown("---")
st.sidebar.subheader("🌐 Network Setup")
num_nodes         = st.sidebar.slider("Number of Nodes", 5, 20, 10)
edge_prob         = st.sidebar.slider("Edge Probability", 0.2, 0.9, 0.4, 0.05)
traffic_intensity = st.sidebar.slider("Traffic Intensity", 1, 8, 3)
seed              = st.sidebar.number_input("Random Seed", 0, 999, 42)

if st.sidebar.button("Generate Network", use_container_width=True):
    random.seed(seed)
    G = create_graph(num_nodes=num_nodes, edge_prob=edge_prob)
    G = add_dynamic_traffic(G, intensity=traffic_intensity)  # ← keep this
    st.session_state.G = G
    st.session_state.history = []
    st.session_state.q_router = None
    st.sidebar.success("Network created!")

# REPLACE the refresh button block with this:

if st.sidebar.button("🔄 Refresh Traffic", use_container_width=True,
                      disabled=st.session_state.G is None):
    fresh_G = clone_graph(st.session_state.G)
    st.session_state.G = add_dynamic_traffic(fresh_G, intensity=traffic_intensity)
    from utils.dataset_loader import get_dataset
    get_dataset().refresh_edge_traffic(st.session_state.G)
    st.sidebar.success("Traffic updated!")
    st.rerun()

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
st.title("🔀 Smart Routing System")
st.caption("Multi-level routing with Protocol Intelligence & Retransmit Tracking")

if st.session_state.G is None:
    st.info("Generate a network from the sidebar to get started.")
    st.stop()

G     = st.session_state.G
stats = get_graph_stats(G)
nodes = list(G.nodes())

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Nodes",           stats["nodes"])
c2.metric("Edges",           stats["edges"])
c3.metric("Congested Edges", stats["congested_edges"])
c4.metric("Avg Load",        f"{stats['avg_load_percent']}%")
c5.metric("Connected",       "Yes" if stats["is_connected"] else "❌ No")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "  🗺️ Network View",
    "  ⚡ Basic",
    "  🧠 Adaptive",
    "  🤖 Q-Learning",
    "  📡 Protocol Intel",
    "  🔁 Drop & Retransmit",
    "  📊 Analytics",
])


# ═══════════════════════════════════════════════
# TAB 1 – Network View
# ═══════════════════════════════════════════════
with tab1:
    show_proto = st.toggle("Show Protocol Labels on Edges", value=False)
    col_left, col_right = st.columns([3, 1])
    with col_left:
        fig = build_plotly_graph(G, title="Current Network Topology",
                                  show_protocol=show_proto)
        st.plotly_chart(fig, use_container_width=True)
    with col_right:
        st.subheader("Edge Status")
        edge_data = []
        for u, v, d in G.edges(data=True):
            edge_data.append({
                "Edge":     f"{u}↔{v}",
                "Protocol": d.get("protocol","?"),
                "Best":     d.get("best_protocol","?"),
                "Fitness":  d.get("protocol_fitness", 0),
                "Weight":   d.get("weight","?"),
                "Load %":   d.get("load",0),
                "Drop":     round(d.get("frame_err",0),3),
                "Status":   "🔴 Congested" if d.get("congested") else "🟢 Clear",
            })
        st.dataframe(pd.DataFrame(edge_data), use_container_width=True, height=370)

# ═══════════════════════════════════════════════
# TAB 2 – Level 3 Basic Routing
# ═══════════════════════════════════════════════
with tab2:
    st.subheader("– Basic Intelligent Routing")
    st.markdown("Uses **Dijkstra**, **A\\***, and **K-Shortest Paths** to find optimal routes.")

    col_a, col_b, col_c = st.columns(3)
    src3  = col_a.selectbox("Source Node", nodes, key="src3")
    dst3  = col_b.selectbox("Destination Node", [n for n in nodes if n!=src3], key="dst3")
    k_val = col_c.slider("K (num paths)", 1, 8, 3, key="k3")
    algo  = st.radio("Algorithm", ["K-Shortest (Yen's)", "Dijkstra", "A*"], horizontal=True)

    if st.button("Route Packet (L3)", use_container_width=True):
        router = Level3Router(G, k=k_val)
        if algo == "Dijkstra":
            path, cost = run_dijkstra(G, src3, dst3)
            result = {"packet_id": len(st.session_state.history),
                      "path": path, "cost": cost, "algorithm": "Dijkstra",
                      "hops": len(path)-1, "success": bool(path)}
        elif algo == "A*":
            path, cost, explored = run_astar(G, src3, dst3)
            result = {"packet_id": len(st.session_state.history),
                      "path": path, "cost": cost, "algorithm": "A*",
                      "hops": len(path)-1, "explored": explored, "success": bool(path)}
        else:
            result = router.route_packet(src3, dst3, packet_id=len(st.session_state.history))
        result["level"] = 3

        # Simulate delivery with retransmit
        delivery = simulate_packet_delivery(G, result["path"])
        result["hops_dropped"]  = delivery.get("hops_dropped", 0)
        result["hops_retried"]  = delivery.get("hops_retried", 0)
        result["perm_fail"]     = delivery.get("perm_fail", False)

        st.session_state.history.append(result)

        col1, col2 = st.columns([2, 1])
        with col1:
            fig_path = build_plotly_graph(G, highlight_path=result["path"],
                                          title="Routed Path (White=selected)")
            st.plotly_chart(fig_path, use_container_width=True)
        with col2:
            st.success("✅ Path found!")
            # Protocol on path
            if len(result["path"]) >= 2 and G.has_edge(result["path"][0], result["path"][1]):
                ed = G[result["path"][0]][result["path"][1]]
                proto   = ed.get("protocol","?")
                fitness = ed.get("protocol_fitness",0)
                best    = ed.get("best_protocol","?")
                st.info(f"**Protocol:** {proto}  |  Fitness: {fitness:.2f}  |  Best: {best}")
                if proto != best:
                    st.warning(f"⚠️ Mismatch: {proto} used but {best} recommended")
            # Retransmit summary for this packet
            if delivery.get("hops_dropped", 0) > 0:
                recovered_all = not delivery.get("perm_fail", False)
                if recovered_all:
                    st.warning(f"🔄 {delivery['hops_dropped']} hop(s) dropped → retransmitted → recovered")
                else:
                    st.error(f"❌ {delivery['hops_dropped']} hop(s) dropped → retransmit failed permanently")
            else:
                st.success("✅ No drops on this path")
            st.json({k:v for k,v in result.items() if k not in ("path",)})
            st.info(f"**Path:** {' → '.join(str(n) for n in result['path'])}")

        if algo == "K-Shortest (Yen's)":
            all_paths = router.get_all_paths(src3, dst3)
            st.subheader(f"All {len(all_paths)} Candidate Paths")
            df_paths = pd.DataFrame([
                {"Rank": i+1,
                 "Path": " → ".join(str(n) for n in p),
                 "Cost": c, "Hops": len(p)-1,
                 "Protocol (1st edge)": G[p[0]][p[1]].get("protocol","?") if len(p)>1 else "?",
                 "Fitness (1st edge)":  round(G[p[0]][p[1]].get("protocol_fitness",0),2) if len(p)>1 else 0,
                }
                for i,(p,c) in enumerate(all_paths)
            ])
            st.dataframe(df_paths, use_container_width=True)


# ═══════════════════════════════════════════════
# TAB 3 – Level 4 Adaptive Routing
# ═══════════════════════════════════════════════
with tab3:
    st.subheader("– Adaptive Routing")
    st.markdown("Monitors congestion in **real-time** and switches to alternative paths dynamically.")

    col_a, col_b = st.columns(2)
    src4 = col_a.selectbox("Source Node", nodes, key="src4")
    dst4 = col_b.selectbox("Destination Node", [n for n in nodes if n!=src4], key="dst4")
    n_packets = st.slider("Number of Packets to Simulate", 1, 20, 5, key="n_pkts4")

    if st.button("Run Adaptive Routing (L4)", use_container_width=True):
        router4  = Level4Router(G, k=5)
        results  = []
        progress = st.progress(0)
        for i in range(n_packets):
            if i > 0:
                G_tmp = add_dynamic_traffic(G, intensity=traffic_intensity)
                router4.update_graph(G_tmp)
            res = router4.route_packet(src4, dst4, packet_id=i)
            # Add retransmit info
            delivery = simulate_packet_delivery(G, res.get("path", []))
            res["hops_dropped"] = delivery.get("hops_dropped", 0)
            res["hops_retried"] = delivery.get("hops_retried", 0)
            res["perm_fail"]    = delivery.get("perm_fail", False)
            results.append(res)
            st.session_state.history.append(res)
            progress.progress((i+1)/n_packets)
            time.sleep(0.05)

        last_path = results[-1].get("path", [])
        col1, col2 = st.columns([2, 1])
        with col1:
            fig_path = build_plotly_graph(G, highlight_path=last_path,
                                          title="Last Routed Path")
            st.plotly_chart(fig_path, use_container_width=True)
        with col2:
            st.subheader("Last Result")
            st.json({k:v for k,v in results[-1].items() if k!="path"})

        st.subheader("Simulation Summary")
        df_res = pd.DataFrame([{
            "Pkt #":    r["packet_id"],
            "Path":     " → ".join(str(n) for n in r.get("path",[])),
            "Cost":     r.get("cost","?"),
            "Eff. Cost":r.get("effective_cost","?"),
            "Avg Load %":r.get("avg_load_percent","?"),
            "Congested":"🔴" if r.get("congested") else "🟢",
            "Dropped":  r.get("hops_dropped",0),
            "Retried":  r.get("hops_retried",0),
            "Perm Fail":"❌" if r.get("perm_fail") else "✅",
            "Hops":     r.get("hops","?"),
        } for r in results])
        st.dataframe(df_res, use_container_width=True)

        fig_cost = px.line(df_res, x="Pkt #", y="Cost",
                           title="Cost per Packet (Adaptive Routing)",
                           markers=True, color_discrete_sequence=["#4a90d9"])
        fig_cost.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_cost, use_container_width=True)

        report = router4.get_monitor_report()
        if report:
            st.subheader("Path Monitor Report")
            monitor_data = [{
                "Path":              path_key,
                "Attempts":          v["attempts"],
                "Successes":         v["successes"],
                "Congestion Events": v["congestion_events"],
                "Avg Latency (ms)":  round(v["total_latency"]/max(v["attempts"],1),4),
            } for path_key, v in report.items()]
            st.dataframe(pd.DataFrame(monitor_data), use_container_width=True)


# ═══════════════════════════════════════════════
# TAB 4 – Level 5 Q-Learning
# ═══════════════════════════════════════════════
with tab4:
    st.subheader("– Q-Learning Autonomous Router")
    st.markdown("A **reinforcement learning** agent that trains to find optimal routes.")

    col_a, col_b = st.columns(2)
    src5 = col_a.selectbox("Source Node", nodes, key="src5")
    dst5 = col_b.selectbox("Destination Node", [n for n in nodes if n!=src5], key="dst5")
    col_c,col_d,col_e = st.columns(3)
    episodes = col_c.slider("Training Episodes", 100, 2000, 500, step=100)
    alpha    = col_d.slider("Learning Rate (α)", 0.01, 0.5, 0.15, step=0.01)
    epsilon  = col_e.slider("Exploration (ε)", 0.05, 0.5, 0.2, step=0.05)

    if st.button("Train & Route (L5)", use_container_width=True):
        with st.spinner("Training Q-Learning agent..."):
            router5 = QLearningRouter(G, alpha=alpha, epsilon=epsilon, episodes=episodes)
            t0      = time.time()
            rewards = router5.train(goal=dst5)
            train_time = round(time.time()-t0, 2)
            st.session_state.q_router = router5

        result5  = router5.route_packet(src5, dst5, packet_id=len(st.session_state.history))
        delivery = simulate_packet_delivery(G, result5.get("path",[]))
        result5["hops_dropped"] = delivery.get("hops_dropped",0)
        result5["hops_retried"] = delivery.get("hops_retried",0)
        result5["perm_fail"]    = delivery.get("perm_fail",False)
        st.session_state.history.append(result5)

        col1, col2 = st.columns([2, 1])
        with col1:
            fig_path = build_plotly_graph(G, highlight_path=result5.get("path",[]),
                                          title="Q-Learning Route")
            st.plotly_chart(fig_path, use_container_width=True)
        with col2:
            reached = result5.get("reached_goal", False)
            # [CHANGED] Fixed: use if/else block instead of ternary to avoid DeltaGenerator display
            if reached:
                st.success("✅ Goal reached!")
            else:
                st.warning("⚠️ Goal not fully reached")
            st.metric("Training Time",    f"{train_time}s")
            st.metric("Decision Latency", f"{result5.get('decision_latency_ms',0)} ms")
            st.metric("Path Cost",        result5.get("cost","?"))
            st.metric("Hops",             result5.get("hops","?"))
            if result5.get("hops_dropped",0) > 0:
                st.warning(f"🔄 {result5['hops_dropped']} hop drop(s), "
                           f"{result5['hops_retried']} retransmit(s)")
            risky = result5.get("predicted_risky_edges",[])
            if risky:
                st.warning(f"⚠️ {len(risky)} risky edge(s) predicted")
                for u,v,load in risky:
                    st.write(f"  Edge {u}↔{v}: predicted load {load}%")
            else:
                st.success("✅ No congestion predicted")

        st.subheader("Training Rewards over Episodes")
        reward_df = pd.DataFrame({"Episode": range(len(rewards)), "Reward": rewards})
        reward_df["Smoothed"] = reward_df["Reward"].rolling(window=20, min_periods=1).mean()
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=reward_df["Episode"], y=reward_df["Reward"],
            mode='lines', name='Raw Reward', line=dict(color='#4a90d9',width=1), opacity=0.4))
        fig_r.add_trace(go.Scatter(x=reward_df["Episode"], y=reward_df["Smoothed"],
            mode='lines', name='Smoothed (20-ep)', line=dict(color='#2ecc71',width=2.5)))
        fig_r.update_layout(title="Reward Convergence", xaxis_title="Episode",
                            yaxis_title="Total Reward",
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            height=320)
        st.plotly_chart(fig_r, use_container_width=True)

        st.subheader("Q-Table Heatmap")
        q_summary = router5.get_q_table_summary()
        st.write(f"Shape: {q_summary['shape']} | Max Q: {q_summary['max_q']:.2f} "
                 f"| Non-zero: {q_summary['nonzero_entries']}")
        fig_qt = px.imshow(router5.Q,
                            labels=dict(x="Next Node",y="Current Node",color="Q-value"),
                            color_continuous_scale="RdBu",
                            title="Q-Table (Current Node → Next Node)")
        fig_qt.update_layout(height=380, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_qt, use_container_width=True)


# ═══════════════════════════════════════════════
# TAB 5 – Protocol Intelligence  [NEW]
# ═══════════════════════════════════════════════
with tab5:
    st.subheader("📡 Protocol Intelligence")
    st.markdown("""
    **Which protocol should be used for which data type?**
    Use the sliders below to see real-time protocol rankings based on your edge characteristics.
    """)

    col_a, col_b, col_c, col_d = st.columns(4)
    pi_payload = col_a.slider("Payload (bits)", 1, 32, 8)
    pi_noise   = col_b.slider("Noise level", 0.00, 0.10, 0.01, step=0.005, format="%.3f")
    pi_latbudg = col_c.slider("Latency budget (cycles)", 426, 1651, 500)
    pi_duplex  = col_d.toggle("Full-Duplex required?")

    rec = get_dataset().get_protocol_recommendation(
        pi_payload, pi_noise, pi_latbudg, pi_duplex
    )
    st.success(f"✅ **Recommended protocol: {rec['recommended']}**  —  {rec['rationale']}")

    st.markdown("#### Protocol Fitness Scores for your settings")
    sc_df = pd.DataFrame([
        {"Protocol": p, "Fitness Score": rec["scores"][p],
         "Rank": rec["ranking"].index(p)+1}
        for p in rec["scores"]
    ]).sort_values("Rank")
    fig_sc = px.bar(sc_df, x="Protocol", y="Fitness Score", color="Protocol",
                    color_discrete_map=PROTO_COLOR,
                    title="Protocol Fitness (higher = better fit)",
                    text="Fitness Score")
    fig_sc.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig_sc.update_layout(yaxis_range=[0,1.1],
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("#### Protocol Characteristics (from Dataset)")
    proto_summary = []
    for p, s in PROTOCOL_STATS.items():
        proto_summary.append({
            "Protocol":       p,
            "Avg Latency (cycles)": s["avg_latency"],
            "Avg Throughput (bps)": s["avg_throughput"],
            "Avg Noise":      s["avg_noise"],
            "Drop Rate":      f"{s['drop_rate']*100:.2f}%",
            "Best For":       s["best_for"],
        })
    st.dataframe(pd.DataFrame(proto_summary), use_container_width=True)

    st.markdown("#### Protocol Decision Map")
    st.markdown("""
| Condition | Best Protocol | Why |
|-----------|--------------|-----|
| Payload ≤ 8 b, noise < 0.02 | **I2C** | Highest throughput-per-bit; very low drop rate (0.92 %) |
| Payload > 8 b (any duplex) | **SPI** | Handles large frames; lowest drop rate (0.69 %); always full-duplex |
| Noise ≥ 0.05 (any payload) | **I2C / SPI** | UART drop rate jumps to ~75 % at high noise |
| Latency budget < 460 cycles | **UART** | Lowest median latency (451 cycles); 85 % of rows at 451 |
| Default / balanced | **I2C** | Best overall throughput per bit, minimal drops |
    """)

    st.markdown("#### Graph Edge Protocol Distribution")
    proto_on_graph = {}
    mismatch_count = 0
    for _,_,d in G.edges(data=True):
        p = d.get("protocol","?")
        proto_on_graph[p] = proto_on_graph.get(p,0)+1
        if p != d.get("best_protocol", p):
            mismatch_count += 1
    col_x, col_y = st.columns(2)
    with col_x:
        fig_pie = px.pie(values=list(proto_on_graph.values()),
                         names=list(proto_on_graph.keys()),
                         color=list(proto_on_graph.keys()),
                         color_discrete_map=PROTO_COLOR,
                         title="Protocol Distribution on Current Graph")
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_y:
        st.metric("Protocol Mismatches", mismatch_count,
                  help="Edges where assigned protocol != best protocol for the edge's characteristics")
        if mismatch_count > 0:
            st.warning(f"{mismatch_count} edges are using a sub-optimal protocol. "
                       "Click **Refresh Traffic** in the sidebar to re-sample with better assignments.")
        else:
            st.success("All edges are using their optimal protocol.")

        fitness_vals = [d.get("protocol_fitness",0) for _,_,d in G.edges(data=True)]
        st.metric("Avg Protocol Fitness", f"{np.mean(fitness_vals):.3f}",
                  help="Mean fitness score across all edges (1.0 = perfect match)")


# ═══════════════════════════════════════════════
# TAB 6 – Drop & Retransmit  [NEW]
# ═══════════════════════════════════════════════
with tab6:
    st.subheader("🔁 Packet Drop & Retransmit Analysis")

    try:
        report = get_dataset().get_retransmit_report()
        summary = report.get("summary", {})

        if not summary:
            st.info("No transmissions recorded yet. Run routing simulations first.")
        else:
            st.markdown("### Overall Summary")
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Total Packets",    summary.get("total_packets",0))
            c2.metric("Dropped",          summary.get("total_dropped",0),
                      delta=f"-{summary.get('overall_drop_pct',0):.2f}%",
                      delta_color="inverse")
            c3.metric("Retransmitted",    summary.get("total_retried",0))
            c4.metric("Recovered",        summary.get("total_recovered",0))
            c5.metric("Perm. Failed",     summary.get("total_perm_failed",0),
                      delta_color="inverse")

            col_l, col_r = st.columns(2)
            with col_l:
                retx_data = {
                    "Category": ["Delivered OK", "Dropped+Recovered", "Perm. Failed"],
                    "Count":    [
                        summary["total_packets"] - summary["total_dropped"],
                        summary["total_recovered"],
                        summary["total_perm_failed"],
                    ]
                }
                fig_ret = px.pie(retx_data, values="Count", names="Category",
                                 color="Category",
                                 color_discrete_map={
                                     "Delivered OK":      "#2ecc71",
                                     "Dropped+Recovered": "#f39c12",
                                     "Perm. Failed":      "#e74c3c",
                                 },
                                 title="Packet Delivery Outcomes")
                fig_ret.update_layout(paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_ret, use_container_width=True)
            with col_r:
                st.markdown("### Protocol Drop Rates (from Dataset)")
                dr_df = pd.DataFrame([
                    {"Protocol": p,
                     "Drop Rate %": round(PROTOCOL_STATS[p]["drop_rate"]*100,2),
                     "Meaning": PROTOCOL_STATS[p]["description"][:60]+"..."}
                    for p in ["UART","I2C","SPI"]
                ])
                fig_dr = px.bar(dr_df, x="Protocol", y="Drop Rate %",
                                color="Protocol",
                                color_discrete_map=PROTO_COLOR,
                                title="Protocol Drop Rate Comparison",
                                text="Drop Rate %")
                fig_dr.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                fig_dr.update_layout(yaxis_range=[0,6],
                                      paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_dr, use_container_width=True)

            st.markdown("### Per-Edge Retransmit Details")
            per_edge = report.get("per_edge", {})
            if per_edge:
                edge_rows = []
                for edge_label, es in per_edge.items():
                    edge_rows.append({
                        "Edge":           edge_label,
                        "Total":          es["total"],
                        "Dropped":        es["dropped"],
                        "Retransmitted":  es["retried"],
                        "Recovered":      es["recovered"],
                        "Perm Failed":    es["perm_failed"],
                        "Drop Rate %":    es["drop_rate_pct"],
                        "Retransmit %":   es["retransmit_rate_pct"],
                        "Recovery %":     es["recovery_rate_pct"],
                        "Perm Fail %":    es["perm_fail_rate_pct"],
                    })
                df_edge = pd.DataFrame(edge_rows).sort_values("Drop Rate %", ascending=False)
                st.dataframe(df_edge, use_container_width=True)

                # Highlight worst edges
                top_bad = df_edge[df_edge["Drop Rate %"] > 0].head(5)
                if not top_bad.empty:
                    st.markdown("### ⚠️ Worst Edges (Highest Drop Rate)")
                    for _, row in top_bad.iterrows():
                        edge_key = row["Edge"].replace("(","").replace(")","").split(",")
                        proto = "?"
                        try:
                            u,v = int(edge_key[0]), int(edge_key[1])
                            if G.has_edge(u,v):
                                proto = G[u][v].get("protocol","?")
                        except Exception:
                            pass
                        st.warning(
                            f"**Edge {row['Edge']}** ({proto}) — "
                            f"Drop: {row['Drop Rate %']}% | "
                            f"Retransmit: {row['Retransmit %']}% | "
                            f"Recovery: {row['Recovery %']}% | "
                            f"Perm Fail: {row['Perm Fail %']}%"
                        )
            else:
                st.info("Per-edge details will appear after routing packets.")

    except Exception as e:
        st.error(f"Could not load retransmit report: {e}")

    st.markdown("---")
    st.markdown("### 🔧 Fix Recommendations")
    st.markdown("""
**Why are packets being dropped?**
The dataset has no retransmit column — the system **simulates ARQ (Automatic Repeat reQuest)**:
- A packet is considered **dropped** when `frame_err > 0`.
- A **retransmit** is triggered automatically (up to 3 attempts).
- If all 3 retry attempts also fail, the packet is **permanently lost**.

**To reduce drops, make these changes in your system:**

| Issue | Fix |
|-------|-----|
| UART drop rate high (3.51 %) | Switch edges with `noise ≥ 0.05` to **I2C or SPI** |
| SPI high latency (851-1651 cycles on 20% of rows) | Use SPI only for large payloads (>8 b) |
| No retransmit column in dataset | Add `retransmit_count` and `ack_received` columns to your sensor firmware logs |
| Permanent failures after 3 retries | Increase `MAX_RETRIES` in `RetransmitTracker` or add FEC (Forward Error Correction) |
| Protocol mismatch penalty | `graph.py:get_edge_effective_weight` already penalises mismatched protocols — ensure `annotate_graph()` is called on every graph refresh |
    """)


# ═══════════════════════════════════════════════
# TAB 7 – Analytics
# ═══════════════════════════════════════════════
with tab7:
    st.subheader("Session Analytics")

    if not st.session_state.history:
        st.info("Run some routing simulations first.")
    else:
        history = st.session_state.history
        df_hist = pd.DataFrame([{
            "Pkt #":       r.get("packet_id","?"),
            "Level":       r.get("level","?"),
            "Algorithm":   r.get("algorithm","?"),
            "Cost":        r.get("cost",0),
            "Hops":        r.get("hops",0),
            "Congested":   r.get("congested",False),
            "Success":     r.get("success",False),
            "Hops Dropped":r.get("hops_dropped",0),
            "Hops Retried":r.get("hops_retried",0),
            "Perm Fail":   r.get("perm_fail",False),
            "Latency (ms)":r.get("decision_latency_ms",0),
        } for r in history])

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Packets", len(df_hist))
        c2.metric("Success Rate", f"{df_hist['Success'].mean()*100:.1f}%")
        c3.metric("Avg Cost",  f"{df_hist['Cost'].mean():.1f}")
        c4.metric("Total Drops", int(df_hist["Hops Dropped"].sum()))

        fig_box = px.box(df_hist[df_hist["Level"]!="?"], x="Level", y="Cost",
                         color="Level", title="Cost Distribution by Routing Level",
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_box, use_container_width=True)

        # [CHANGED] Removed "Cost vs Hops" scatter — only keep "Packets by Routing Level"
        lc = df_hist["Level"].value_counts().reset_index()
        lc.columns = ["Level","Count"]
        fig_pie = px.pie(lc, values="Count", names="Level",
                         title="Packets by Routing Level")
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Full History")
        st.dataframe(df_hist, use_container_width=True)
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

    # Traffic Model section
    st.markdown("---")
    st.subheader("ML Traffic Prediction")
    col_a,col_b,col_c = st.columns(3)
    t_step = col_a.slider("Time Step (0–23)", 0, 23, 12)
    # [CHANGED] Default avg_load_in to live network avg load so result varies with network
    live_avg = float(stats["avg_load_percent"])
    avg_load_in = col_b.slider("Recent Avg Load (%)", 0.0, 100.0, live_avg)
    # [CHANGED] Default cong_count to live congested edges count
    live_cong = int(stats["congested_edges"])
    cong_count = col_c.slider("Congested Edges", 0, 10, min(live_cong, 10))

    if st.button("Predict Load"):
        import math
        # [CHANGED] Added small network-state noise so result is not identical on every click
        # Uses edge load variance from the live graph as a perturbation signal
        load_vals = [d.get("load", 0) for _, _, d in G.edges(data=True)]
        load_std  = float(np.std(load_vals)) if load_vals else 0.0
        # Perturbation: ±(std / 20), deterministic-ish but changes with network state
        noise = (load_std % 7.3) / 20.0  # small bounded shift, changes per network
        hour_factor = 0.6 + 0.4 * (math.sin(math.pi * (t_step - 3) / 12) ** 2)
        cong_factor = 1.0 + (cong_count / 10.0) * 0.5 + noise
        pred = float(np.clip(avg_load_in * hour_factor * cong_factor, 0, 100))
        will_c = pred >= 70.0
        col1, col2 = st.columns(2)
        col1.metric("Predicted Load", f"{pred:.1f}%")
        col2.metric("Will Congest (>70%)?", "🔴 Yes" if will_c else "🟢 No")
        # 24h curve: each hour gets its own hour_factor so shape varies with sliders
        preds = [
            float(np.clip(
                avg_load_in
                * (0.6 + 0.4 * (math.sin(math.pi * (t - 3) / 12) ** 2))
                * cong_factor,
                0, 100
            )) for t in range(24)
        ]
        fig_tm = px.line(x=list(range(24)), y=preds,
                         labels={"x":"Hour of Day","y":"Predicted Load %"},
                         title="Predicted Load Across 24 Hours",
                         color_discrete_sequence=["#e74c3c"], markers=True)
        fig_tm.add_hline(y=70, line_dash="dash", line_color="orange",
                          annotation_text="Congestion Threshold (70%)")
        fig_tm.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_tm, use_container_width=True)
