"""
tests/test_routing.py
----------------------
Unit tests for all routing levels.
Run with: python -m pytest tests/test_routing.py -v
       or: python tests/test_routing.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random
import unittest

from core.graph import create_graph, add_dynamic_traffic, get_graph_stats
from core.simulator import calculate_cost, is_path_congested, simulate_packet_delivery
from level3.basic_routing import Level3Router, get_k_shortest_paths, run_dijkstra, run_astar
from level4.adaptive_routing import Level4Router, PathMonitor
from level5.qlearning_router import QLearningRouter, CongestionPredictor
from utils.traffic_model import TrafficModel


class TestGraph(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        self.G = create_graph(num_nodes=8, edge_prob=0.5)

    def test_connected(self):
        import networkx as nx
        self.assertTrue(nx.is_connected(self.G))

    def test_nodes(self):
        self.assertEqual(self.G.number_of_nodes(), 8)

    def test_dynamic_traffic(self):
        G2 = add_dynamic_traffic(self.G, intensity=2)
        for _, _, d in G2.edges(data=True):
            self.assertGreaterEqual(d['weight'], 1)

    def test_stats(self):
        stats = get_graph_stats(self.G)
        self.assertIn('nodes', stats)
        self.assertEqual(stats['nodes'], 8)


class TestSimulator(unittest.TestCase):
    def setUp(self):
        random.seed(1)
        self.G = create_graph(num_nodes=6, edge_prob=0.6)
        self.nodes = list(self.G.nodes())

    def test_calculate_cost(self):
        import networkx as nx
        path = nx.dijkstra_path(self.G, self.nodes[0], self.nodes[-1], weight='weight')
        cost = calculate_cost(self.G, path)
        self.assertGreater(cost, 0)

    def test_simulate_delivery(self):
        import networkx as nx
        path = nx.dijkstra_path(self.G, self.nodes[0], self.nodes[-1], weight='weight')
        result = simulate_packet_delivery(self.G, path, packet_id=99)
        self.assertIn('cost', result)
        self.assertIn('hops', result)


class TestLevel3(unittest.TestCase):
    def setUp(self):
        random.seed(2)
        self.G = create_graph(num_nodes=10, edge_prob=0.45)
        self.nodes = list(self.G.nodes())
        self.src, self.dst = self.nodes[0], self.nodes[-1]

    def test_dijkstra(self):
        path, cost = run_dijkstra(self.G, self.src, self.dst)
        self.assertIsInstance(path, list)
        self.assertGreater(cost, 0)
        self.assertEqual(path[0], self.src)
        self.assertEqual(path[-1], self.dst)

    def test_astar(self):
        path, cost, explored = run_astar(self.G, self.src, self.dst)
        self.assertIsInstance(path, list)
        self.assertGreater(explored, 0)

    def test_k_shortest(self):
        paths = get_k_shortest_paths(self.G, self.src, self.dst, k=3)
        self.assertGreater(len(paths), 0)
        costs = [c for _, c in paths]
        self.assertEqual(costs, sorted(costs))  # must be sorted ascending

    def test_router(self):
        router = Level3Router(self.G, k=4)
        result = router.route_packet(self.src, self.dst, packet_id=1)
        self.assertTrue(result['success'])
        self.assertEqual(result['level'], 3)
        self.assertIn('path', result)


class TestLevel4(unittest.TestCase):
    def setUp(self):
        random.seed(3)
        self.G = create_graph(num_nodes=10, edge_prob=0.5)
        self.G = add_dynamic_traffic(self.G, intensity=3)
        self.nodes = list(self.G.nodes())
        self.src, self.dst = self.nodes[0], self.nodes[-1]

    def test_path_monitor(self):
        pm = PathMonitor()
        path = [0, 1, 2]
        pm.record(path, success=True, latency=1.0)
        pm.record(path, success=False, latency=2.0)
        self.assertAlmostEqual(pm.success_rate(path), 0.5)

    def test_router(self):
        router = Level4Router(self.G, k=4)
        result = router.route_packet(self.src, self.dst, packet_id=10)
        self.assertIn('level', result)
        self.assertEqual(result['level'], 4)
        self.assertIn('effective_cost', result)

    def test_graph_update(self):
        router = Level4Router(self.G, k=4)
        G2 = add_dynamic_traffic(self.G, intensity=5)
        router.update_graph(G2)  # should not crash
        result = router.route_packet(self.src, self.dst)
        self.assertIn('path', result)


class TestLevel5(unittest.TestCase):
    def setUp(self):
        random.seed(4)
        self.G = create_graph(num_nodes=8, edge_prob=0.55)
        self.G = add_dynamic_traffic(self.G, intensity=2)
        self.nodes = list(self.G.nodes())
        self.src, self.dst = self.nodes[0], self.nodes[-1]

    def test_train(self):
        router = QLearningRouter(self.G, episodes=200)
        history = router.train(self.dst)
        self.assertEqual(len(history), 200)

    def test_get_path(self):
        router = QLearningRouter(self.G, episodes=300)
        router.train(self.dst)
        path = router.get_path(self.src, self.dst)
        self.assertIsInstance(path, list)
        self.assertGreater(len(path), 0)
        self.assertEqual(path[0], self.src)

    def test_route_packet(self):
        router = QLearningRouter(self.G, episodes=300)
        result = router.route_packet(self.src, self.dst, packet_id=20)
        self.assertEqual(result['level'], 5)
        self.assertIn('predicted_risky_edges', result)

    def test_congestion_predictor(self):
        cp = CongestionPredictor(window=5)
        for load in [60, 70, 80, 85, 90]:
            cp.record(0, 1, load)
        pred = cp.predict(0, 1)
        self.assertGreater(pred, 60)
        self.assertTrue(cp.will_congest(0, 1, threshold=70))


class TestTrafficModel(unittest.TestCase):
    def test_predict(self):
        tm = TrafficModel()
        tm.train()
        pred = tm.predict_load(12, 50.0, 2)
        self.assertGreaterEqual(pred, 0)
        self.assertLessEqual(pred, 100)

    def test_will_congest(self):
        tm = TrafficModel()
        tm.train()
        result = tm.will_congest(12, 80.0, 5, threshold=60.0)
        self.assertIsInstance(result, bool)


if __name__ == '__main__':
    print("Running Smart Routing System Tests...\n")
    unittest.main(verbosity=2)
