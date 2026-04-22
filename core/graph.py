"""
core/graph.py  [UPGRADED]
--------------------------
Enhanced NoC graph creation and management.
UPGRADES:
  - get_edge_effective_weight now penalises mismatched protocols
    (if the assigned protocol is not the best fit, cost is raised).
  - get_graph_stats now includes per-protocol drop/retransmit summary.
"""

import networkx as nx
import random
import copy


def create_graph(num_nodes: int = 10, edge_prob: float = 0.4) -> nx.Graph:
    G = nx.Graph()
    for i in range(num_nodes):
        G.add_node(i, pos=(random.uniform(0, 10), random.uniform(0, 10)))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_prob:
                weight = random.randint(1, 10)
                G.add_edge(i, j, weight=weight, load=0, congested=False,
                           latency_cycles=451.0, throughput_bps=39988.0,
                           noise=0.03, frame_err=0.0, protocol="I2C",
                           protocol_fitness=1.0, best_protocol="I2C",
                           payload_bits=8.0)
    if not nx.is_connected(G):
        nodes = list(G.nodes())
        for i in range(len(nodes) - 1):
            if not nx.has_path(G, nodes[i], nodes[i + 1]):
                G.add_edge(nodes[i], nodes[i + 1],
                           weight=random.randint(1, 10), load=0, congested=False,
                           latency_cycles=451.0, throughput_bps=39988.0,
                           noise=0.03, frame_err=0.0, protocol="I2C",
                           protocol_fitness=1.0, best_protocol="I2C",
                           payload_bits=8.0)
    from utils.dataset_loader import get_dataset
    G = get_dataset().annotate_graph(G)

    return G


# REPLACE the entire add_dynamic_traffic function with this:

def add_dynamic_traffic(G: nx.Graph, intensity: int = 3) -> nx.Graph:
    for u, v, data in G.edges(data=True):
        change = random.randint(-intensity, intensity)
        data['weight']    = max(1, data['weight'] + change)
        data['load']      = random.randint(0, 100)
        data['congested'] = data['load'] > 75
    return G


def get_edge_effective_weight(G: nx.Graph, u, v) -> float:
    data      = G[u][v]
    base      = data.get('weight', 1)

    if data.get('congested', False):
        return base * 3

    load_factor = 1 + data.get('load', 0) / 100.0
    noise_penalty = data.get('noise', 0) * 5
    ferr_penalty  = data.get('frame_err', 0) * 10

    # Protocol mismatch penalty: if assigned protocol != best, add cost
    assigned  = data.get('protocol', 'UART')
    best      = data.get('best_protocol', assigned)
    fitness   = data.get('protocol_fitness', 0.7)
    mismatch_penalty = (1.0 - fitness) * 2.0   # up to +2 for worst mismatch

    return base * load_factor + noise_penalty + ferr_penalty + mismatch_penalty


def clone_graph(G: nx.Graph) -> nx.Graph:
    return copy.deepcopy(G)


def get_graph_stats(G: nx.Graph) -> dict:
    congested_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('congested', False)]
    avg_load = (
        sum(d.get('load', 0) for _, _, d in G.edges(data=True)) / G.number_of_edges()
        if G.number_of_edges() > 0 else 0
    )
    protocol_counts = {}
    for _, _, d in G.edges(data=True):
        p = d.get('protocol', 'UNKNOWN')
        protocol_counts[p] = protocol_counts.get(p, 0) + 1

    avg_latency = (
        sum(d.get('latency_cycles', 0) for _, _, d in G.edges(data=True)) / G.number_of_edges()
        if G.number_of_edges() > 0 else 0
    )

    # Collect retransmit/drop info from dataset tracker
    retransmit_summary = {}
    try:
        from utils.dataset_loader import get_dataset
        retransmit_summary = get_dataset().get_retransmit_report()["summary"]
    except Exception:
        pass

    return {
        "nodes":                 G.number_of_nodes(),
        "edges":                 G.number_of_edges(),
        "congested_edges":       len(congested_edges),
        "avg_load_percent":      round(avg_load, 2),
        "avg_latency_cycles":    round(avg_latency, 2),
        "protocol_distribution": protocol_counts,
        "is_connected":          nx.is_connected(G),
        "retransmit_summary":    retransmit_summary,
    }
