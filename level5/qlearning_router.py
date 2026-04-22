"""
level5/qlearning_router.py
---------------------------
Level 5 – Autonomous Intelligent Routing (Q-Learning).

Self-learning router that:
- Learns from past routing decisions and congestion patterns.
- Improves path selection over time through Q-learning.
- Predicts congestion using a lightweight traffic model.
- Integrates with Level 4 adaptive routing as a fallback.
"""

import numpy as np
import random
import time
import networkx as nx
from typing import List, Optional, Dict, Tuple
from collections import deque

from core.simulator import is_path_congested, path_avg_load, calculate_cost
from level4.adaptive_routing import Level4Router


# ──────────────────────────────────────────────────────────────────────────────
# Congestion Predictor (lightweight linear model)
# ──────────────────────────────────────────────────────────────────────────────

class CongestionPredictor:
    """
    Simple online linear model to predict future edge load.
    Trained incrementally on observed load history.
    """

    def __init__(self, window: int = 10):
        self.window = window
        self.history: Dict[Tuple, deque] = {}

    def record(self, u, v, load: float):
        key = (min(u, v), max(u, v))
        if key not in self.history:
            self.history[key] = deque(maxlen=self.window)
        self.history[key].append(load)

    def predict(self, u, v) -> float:
        """Predict next load based on exponential moving average."""
        key = (min(u, v), max(u, v))
        hist = self.history.get(key)
        if not hist:
            return 50.0  # default unknown = moderate
        arr = np.array(list(hist))
        weights = np.exp(np.linspace(0, 1, len(arr)))
        weights /= weights.sum()
        return float(np.dot(weights, arr))

    def will_congest(self, u, v, threshold: float = 70.0) -> bool:
        return self.predict(u, v) >= threshold


# ──────────────────────────────────────────────────────────────────────────────
# Q-Learning Router
# ──────────────────────────────────────────────────────────────────────────────

class QLearningRouter:
    """
    Q-Learning based autonomous router.

    State  : current node index
    Action : next hop (neighbor index)
    Reward : -effective_edge_cost (maximise negative cost = minimise cost)
             + bonus for reaching goal
             + penalty for visiting congested edges
    """

    def __init__(
        self,
        G: nx.Graph,
        alpha: float = 0.15,
        gamma: float = 0.9,
        epsilon: float = 0.2,
        episodes: int = 800,
        max_steps: int = 100,
    ):
        self.G = G
        self.nodes = list(G.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        self.idx_to_node = {i: n for i, n in enumerate(self.nodes)}
        self.n = len(self.nodes)

        self.Q = np.zeros((self.n, self.n))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.max_steps = max_steps

        self.rewards_history: List[float] = []
        self.predictor = CongestionPredictor()
        self._trained = False
        self._training_goal: Optional[int] = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def neighbors(self, node) -> List:
        return list(self.G.neighbors(node))

    def _edge_reward(self, u, v, goal) -> float:
        """Reward signal for taking edge u→v toward goal."""
        if not self.G.has_edge(u, v):
            return -100.0

        w = self.G[u][v].get('weight', 1)
        load = self.G[u][v].get('load', 0)
        congested = self.G[u][v].get('congested', False)

        reward = -w
        if congested:
            reward -= 20
        if load > 70:
            reward -= 10
        if v == goal:
            reward += 100  # goal bonus

        # Record for predictor
        self.predictor.record(u, v, load)

        return reward

    def _choose_action(self, node, visited: set) -> Optional:
        """ε-greedy action selection, avoiding already-visited nodes."""
        nbrs = [n for n in self.neighbors(node) if n not in visited]
        if not nbrs:
            return None
        if random.random() < self.epsilon:
            return random.choice(nbrs)
        i = self.node_to_idx[node]
        q_vals = [self.Q[i][self.node_to_idx[n]] for n in nbrs]
        return nbrs[int(np.argmax(q_vals))]

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, goal, episodes_override=None) -> List[float]:
        """
        Train Q-table for routing toward `goal`.
        Uses ε-greedy exploration with decaying epsilon.
        [CHANGED] Removed redundant warm-start branch; unified episode logic.
        """
        # [CHANGED] Always use the configured episodes count; no warm-start shortcut
        episodes = episodes_override if episodes_override else self.episodes
        self._training_goal = goal
        self.rewards_history = []

        for ep in range(episodes):
            current = random.choice(self.nodes)
            visited = {current}
            total_reward = 0.0

            # Decay exploration over time (eps computed cleanly per episode)
            eps = max(0.05, self.epsilon * (1 - ep / max(episodes, 1)))

            for _ in range(self.max_steps):
                if current == goal:
                    break

                nbrs = [n for n in self.neighbors(current) if n not in visited]
                if not nbrs:
                    break

                # ε-greedy
                if random.random() < eps:
                    nxt = random.choice(nbrs)
                else:
                    i = self.node_to_idx[current]
                    q_vals = [self.Q[i][self.node_to_idx[n]] for n in nbrs]
                    nxt = nbrs[int(np.argmax(q_vals))]

                r = self._edge_reward(current, nxt, goal)
                total_reward += r

                i = self.node_to_idx[current]
                j = self.node_to_idx[nxt]
                best_next_q = np.max(self.Q[j])
                self.Q[i][j] += self.alpha * (r + self.gamma * best_next_q - self.Q[i][j])

                visited.add(nxt)
                current = nxt

            self.rewards_history.append(total_reward)

        self._trained = True
        return self.rewards_history

    # ── Inference ─────────────────────────────────────────────────────────────

    def get_path(self, start, goal) -> List:
        """
        Greedily follow Q-values from start to goal.
        Avoids cycles using a visited set.
        """
        path = [start]
        current = start
        visited = {current}

        for _ in range(self.max_steps):
            if current == goal:
                break

            nbrs = [n for n in self.neighbors(current) if n not in visited]
            if not nbrs:
                break

            i = self.node_to_idx[current]
            q_vals = [self.Q[i][self.node_to_idx[n]] for n in nbrs]
            nxt = nbrs[int(np.argmax(q_vals))]

            path.append(nxt)
            visited.add(nxt)
            current = nxt

        # Append goal if reachable directly
        if path[-1] != goal and self.G.has_edge(path[-1], goal):
            path.append(goal)

        return path

    def predict_congestion(self, path: List) -> List[Tuple]:
        """
        Predict which edges on the path are likely to congest soon.
        Returns list of (u, v, predicted_load) for risky edges.
        """
        risky = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            pred = self.predictor.predict(u, v)
            if pred >= 65:
                risky.append((u, v, round(pred, 1)))
        return risky

    # ── Routing Interface ─────────────────────────────────────────────────────

    def route_packet(self, start, end, packet_id: int = 0) -> dict:
        """
        Route a packet using learned Q-values.
        Auto-trains if not already trained for this destination.
        """
        t0 = time.perf_counter()

        if not self._trained or self._training_goal != end:
            self.train(goal=end)

        path = self.get_path(start, end)
        latency = (time.perf_counter() - t0) * 1000

        if not path or len(path) < 2:
            return {"packet_id": packet_id, "success": False, "reason": "no_path_learned"}

        cost = calculate_cost(self.G, path)
        congested = is_path_congested(self.G, path)
        avg_load = path_avg_load(self.G, path)
        predicted_risks = self.predict_congestion(path)
        reached_goal = path[-1] == end

        return {
            "packet_id": packet_id,
            "level": 5,
            "algorithm": "Q-Learning (autonomous)",
            "path": path,
            "cost": cost,
            "avg_load_percent": round(avg_load, 2),
            "congested": congested,
            "reached_goal": reached_goal,
            "predicted_risky_edges": predicted_risks,
            "hops": len(path) - 1,
            "decision_latency_ms": round(latency, 4),
            "training_episodes": self.episodes,
            "success": reached_goal,
        }

    def update_graph(self, G: nx.Graph):
        """Update the graph and reset training to force re-learning."""
        self.G = G
        self._trained = False

    def get_q_table_summary(self) -> dict:
        return {
            "shape": self.Q.shape,
            "max_q": float(np.max(self.Q)),
            "min_q": float(np.min(self.Q)),
            "nonzero_entries": int(np.count_nonzero(self.Q)),
        }
