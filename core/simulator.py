"""
core/simulator.py [UPGRADED]
------------------------------
Path cost calculation and simulation utilities.
UPGRADE: simulate_packet_delivery now returns full retransmit details
         (was_dropped, retried, recovered, attempts, final_ok).
"""
import networkx as nx
from typing import List


def calculate_cost(G: nx.Graph, path: List) -> float:
    cost = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        if G.has_edge(u, v):
            cost += G[u][v]['weight']
    return cost


def calculate_effective_cost(G: nx.Graph, path: List) -> float:
    from core.graph import get_edge_effective_weight
    cost = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        if G.has_edge(u, v):
            cost += get_edge_effective_weight(G, u, v)
    return cost


def is_path_congested(G: nx.Graph, path: List) -> bool:
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        if G.has_edge(u, v) and G[u][v].get('congested', False):
            return True
    return False


def path_avg_load(G: nx.Graph, path: List) -> float:
    if len(path) < 2:
        return 0.0
    loads = [G[path[i]][path[i+1]].get('load', 0)
             for i in range(len(path)-1) if G.has_edge(path[i], path[i+1])]
    return sum(loads) / len(loads) if loads else 0.0


def path_avg_latency(G: nx.Graph, path: List) -> float:
    if len(path) < 2:
        return 0.0
    lats = [G[path[i]][path[i+1]].get('latency_cycles', 451.0)
            for i in range(len(path)-1) if G.has_edge(path[i], path[i+1])]
    return round(sum(lats) / len(lats), 2) if lats else 0.0


def simulate_packet_delivery(G: nx.Graph, path: List, packet_id: int = 0) -> dict:
    """
    Simulate delivery of one packet along path.
    Now includes per-hop retransmit simulation via the RetransmitTracker.
    """
    if not path or len(path) < 2:
        return {"packet_id": packet_id, "success": False,
                "reason": "empty_path", "cost": 0}

    congested   = is_path_congested(G, path)
    cost        = calculate_cost(G, path)
    avg_load    = path_avg_load(G, path)
    avg_latency = path_avg_latency(G, path)

    # Simulate retransmit on each hop
    hop_results  = []
    total_drops  = 0
    total_retries = 0
    any_perm_fail = False

    try:
        from utils.dataset_loader import get_dataset
        tracker = get_dataset().retransmit_tracker
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if G.has_edge(u, v):
                frame_err = G[u][v].get('frame_err', 0.0)
                protocol  = G[u][v].get('protocol', 'UART')
                result    = tracker.record_transmission(u, v, frame_err, protocol)
                hop_results.append(result)
                if result["dropped"]:
                    total_drops += 1
                if result["retried"]:
                    total_retries += 1
                if result["dropped"] and not result["recovered"]:
                    any_perm_fail = True
    except Exception:
        pass

    final_success = (not congested) and (not any_perm_fail)

    return {
        "packet_id":         packet_id,
        "path":              path,
        "cost":              cost,
        "avg_load":          round(avg_load, 2),
        "avg_latency_cycles": avg_latency,
        "congested":         congested,
        "hops_dropped":      total_drops,
        "hops_retried":      total_retries,
        "perm_fail":         any_perm_fail,
        "success":           final_success,
        "hops":              len(path) - 1,
    }
