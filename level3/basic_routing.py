"""
level3/basic_routing.py
------------------------
Level 3 – Basic Intelligent Routing.
K-shortest paths, Dijkstra, A* with dataset-aware cost functions.
"""

import networkx as nx
import math
from typing import List, Tuple, Optional
from core.graph import get_edge_effective_weight
from core.simulator import calculate_cost, calculate_effective_cost


def run_dijkstra(G: nx.Graph, src, dst) -> Tuple[List, float]:
    path = nx.dijkstra_path(G, src, dst, weight='weight')
    cost = calculate_cost(G, path)
    return path, cost


def run_astar(G: nx.Graph, src, dst) -> Tuple[List, float, int]:
    pos = nx.get_node_attributes(G, 'pos')
    def heuristic(u, v):
        if pos:
            ux, uy = pos.get(u, (0, 0))
            vx, vy = pos.get(v, (0, 0))
            return math.hypot(ux - vx, uy - vy)
        return 0
    path = nx.astar_path(G, src, dst, heuristic=heuristic, weight='weight')
    cost = calculate_cost(G, path)
    return path, cost, len(path)


def get_k_shortest_paths(G: nx.Graph, src, dst, k: int = 5) -> List[Tuple[List, float]]:
    paths = []
    try:
        for p in nx.shortest_simple_paths(G, src, dst, weight='weight'):
            cost = calculate_cost(G, p)
            paths.append((p, cost))
            if len(paths) >= k:
                break
    except nx.NetworkXNoPath:
        pass
    return sorted(paths, key=lambda x: x[1])


class Level3Router:
    def __init__(self, G: nx.Graph, k: int = 5):
        self.G = G
        self.k = k
        self._cache = {}

    def compute_paths(self, src, dst) -> List[Tuple[List, float]]:
        key = (src, dst)
        if key not in self._cache:
            self._cache[key] = get_k_shortest_paths(self.G, src, dst, self.k)
        return self._cache[key]

    def get_all_paths(self, src, dst) -> List[Tuple[List, float]]:
        return self.compute_paths(src, dst)

    def clear_cache(self):
        self._cache.clear()

    def route_packet(self, src, dst, packet_id: int = 0) -> dict:
        paths = self.compute_paths(src, dst)
        if not paths:
            return {"packet_id": packet_id, "success": False, "reason": "no_path"}
        path, cost = paths[0]
        eff_cost = calculate_effective_cost(self.G, path)
        # Get dataset protocol info from first edge
        proto = "UNKNOWN"
        if len(path) >= 2 and self.G.has_edge(path[0], path[1]):
            proto = self.G[path[0]][path[1]].get("protocol", "UNKNOWN")
        return {
            "packet_id": packet_id,
            "level": 3,
            "algorithm": "K-Shortest (Yen's)",
            "path": path,
            "cost": cost,
            "effective_cost": round(eff_cost, 2),
            "hops": len(path) - 1,
            "protocol": proto,
            "success": True,
        }
