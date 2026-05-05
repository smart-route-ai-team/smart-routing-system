"""
level4/adaptive_routing.py
---------------------------
Level 4 – Adaptive Routing.

Extends Level 3 with real-time path monitoring.
Tracks load, delay, and congestion per edge.
Automatically switches to the next best available path when congestion is detected.
"""

import time
import random
import networkx as nx
from typing import List, Tuple, Optional, Dict

from level3.basic_routing import Level3Router
from core.simulator import is_path_congested, path_avg_load, calculate_effective_cost


# ──────────────────────────────────────────────────────────────────────────────
# Path Monitor
# ──────────────────────────────────────────────────────────────────────────────

class PathMonitor:
    """
    Monitors the health of routing paths in real-time.
    Tracks per-path metrics: congestion events, avg latency, success rate.
    """

    def __init__(self):
        self.stats: Dict[str, dict] = {}

    def _key(self, path: List) -> str:
        return "->".join(str(n) for n in path)

    def record(self, path: List, success: bool, latency: float = 0.0):
        k = self._key(path)
        if k not in self.stats:
            self.stats[k] = {"attempts": 0, "successes": 0, "total_latency": 0.0, "congestion_events": 0}
        self.stats[k]["attempts"] += 1
        if success:
            self.stats[k]["successes"] += 1
        else:
            self.stats[k]["congestion_events"] += 1
        self.stats[k]["total_latency"] += latency

    def success_rate(self, path: List) -> float:
        k = self._key(path)
        s = self.stats.get(k)
        if not s or s["attempts"] == 0:
            return 1.0
        return s["successes"] / s["attempts"]

    def avg_latency(self, path: List) -> float:
        k = self._key(path)
        s = self.stats.get(k)
        if not s or s["attempts"] == 0:
            return 0.0
        return s["total_latency"] / s["attempts"]

    def get_report(self) -> dict:
        return self.stats


# ──────────────────────────────────────────────────────────────────────────────
# Level 4 Router
# ──────────────────────────────────────────────────────────────────────────────

class Level4Router:
    """
    Adaptive router that monitors congestion and switches paths dynamically.

    Strategy:
    1. Compute K-shortest paths (via Level3Router).
    2. Before routing each packet, check current path for congestion.
    3. If congested → switch to the next best non-congested path.
    4. Track all routing decisions via PathMonitor.
    """

    def __init__(self, G: nx.Graph, k: int = 5, congestion_threshold: int = 75):
        self.G = G
        self.k = k
        self.congestion_threshold = congestion_threshold
        self._l3 = Level3Router(G, k=k)
        self.monitor = PathMonitor()
        self._current_path_index: Dict[Tuple, int] = {}  # (src,dst) -> current path rank

    def _score_path(self, path: List, cost: float) -> float:
        """
        Score a path: lower is better.
        Factors in effective cost (with congestion penalty) and monitor history.
        """
        eff_cost = calculate_effective_cost(self.G, path)
        sr = self.monitor.success_rate(path)
        penalty = (1 - sr) * 50  # penalise historically bad paths
        return eff_cost + penalty

    def _rank_paths(self, paths: List[Tuple[List, float]]) -> List[Tuple[List, float]]:
        """Re-rank paths based on live network state and historical performance."""
        scored = [(path, cost, self._score_path(path, cost)) for path, cost in paths]
        scored.sort(key=lambda x: x[2])
        return [(p, c) for p, c, _ in scored]

    def select_path(self, start, end) -> Optional[Tuple[List, float]]:
        """
        Select the best non-congested path.
        Falls back to the least-bad path if all are congested.
        """
        raw_paths = self._l3.compute_paths(start, end)
        if not raw_paths:
            return None

        ranked = self._rank_paths(raw_paths)

        # Try to find a non-congested path
        for path, cost in ranked:
            if not is_path_congested(self.G, path):
                return path, cost

        # All paths congested — return the one with lowest load
        best = min(ranked, key=lambda pc: path_avg_load(self.G, pc[0]))
        return best

    def route_packet(self, start, end, packet_id: int = 0) -> dict:
        """
        Route a packet adaptively with sticky path state.
        Continues from the last used path index; switches only when congested.
        """
        t0 = time.perf_counter()
        key = (start, end)

        raw_paths = self._l3.compute_paths(start, end)
        if not raw_paths:
            return {"packet_id": packet_id, "success": False, "reason": "no_path"}

        ranked = self._rank_paths(raw_paths)
        last_idx = self._current_path_index.get(key, 0)

        selected_path = None
        selected_cost = None
        for i in range(len(ranked)):
            idx = (last_idx + i) % len(ranked)
            path, cost = ranked[idx]
            if not is_path_congested(self.G, path):
                self._current_path_index[key] = idx
                selected_path = path
                selected_cost = cost
                break

        # All paths congested — pick least loaded, remember its index
        if selected_path is None:
            idx, (selected_path, selected_cost) = min(
                enumerate(ranked),
                key=lambda x: path_avg_load(self.G, x[1][0])
            )
            self._current_path_index[key] = idx

        latency = (time.perf_counter() - t0) * 1000
        congested = is_path_congested(self.G, selected_path)
        avg_load = path_avg_load(self.G, selected_path)
        eff_cost = calculate_effective_cost(self.G, selected_path)

        self.monitor.record(selected_path, success=not congested, latency=latency)

        return {
            "packet_id": packet_id,
            "level": 4,
            "algorithm": "adaptive (congestion-aware)",
            "path": selected_path,
            "cost": selected_cost,
            "effective_cost": round(eff_cost, 2),
            "avg_load_percent": round(avg_load, 2),
            "congested": congested,
            "hops": len(selected_path) - 1,
            "decision_latency_ms": round(latency, 4),
            "success": True,
        }

    def update_graph(self, G: nx.Graph):
        """Hot-swap the underlying graph (for live network updates)."""
        self.G = G
        self._l3.G = G
        self._l3.clear_cache()

    def get_monitor_report(self) -> dict:
        return self.monitor.get_report()
