"""
main.py — Smart Routing System – Main Orchestrator
Demonstrates all three levels using REAL protocol dataset metrics.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import random, time
from core.graph import create_graph, add_dynamic_traffic, get_graph_stats
from core.simulator import simulate_packet_delivery
from level3.basic_routing import Level3Router
from level4.adaptive_routing import Level4Router
from level5.qlearning_router import QLearningRouter
from utils.traffic_model import TrafficModel
from utils.dataset_loader import get_dataset
from utils.logger import log_routing_decision, log_network_state

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║       AI-BASED SMART NoC ROUTING SYSTEM  v2.0                   ║
║  Level 3: K-Shortest | Level 4: Adaptive | Level 5: RL          ║
║  Dataset: augmented_protocol_dataset.csv (30,000 real samples)  ║
╚══════════════════════════════════════════════════════════════════╝
"""


def demo_dataset():
    print("\n" + "─"*62)
    print("  DATASET — augmented_protocol_dataset.csv")
    print("─"*62)
    ds = get_dataset()
    s = ds.summary()
    print(f"  Rows      : {s['total_rows']:,}")
    print(f"  Protocols : {s['protocols']}")
    print(f"  Avg Latency: {s['avg_latency_cycles']} cycles")
    print(f"  Avg Throughput: {s['avg_throughput_bps']:.1f} bps")
    print(f"  Avg Noise : {s['avg_noise']}")
    print(f"  Congestion rate: {s['congestion_rate_pct']}%")


def demo_level3(G, src, dst, k=5):
    print("\n" + "─"*62)
    print("  LEVEL 3 — K-Shortest Paths (dataset-weighted edges)")
    print("─"*62)
    router = Level3Router(G, k=k)
    paths = router.get_all_paths(src, dst)
    for rank, (path, cost) in enumerate(paths, 1):
        # Show protocol on first edge
        proto = G[path[0]][path[1]].get("protocol","?") if len(path)>1 else "?"
        print(f"  [{rank}] Cost={cost:5.1f}  Hops={len(path)-1}  Proto={proto}  Path={path}")
    result = router.route_packet(src, dst, packet_id=1)
    log_routing_decision(result)
    return result


def demo_level4(G, src, dst, n_packets=5):
    print("\n" + "─"*62)
    print("  LEVEL 4 — Adaptive Routing (congestion-aware + dataset)")
    print("─"*62)
    router = Level4Router(G, k=5)
    results = []
    for pid in range(n_packets):
        G = add_dynamic_traffic(G, intensity=4)
        router.update_graph(G)
        result = router.route_packet(src, dst, packet_id=pid+10)
        log_routing_decision(result)
        # Show protocol for chosen path
        if result.get('path') and len(result['path']) > 1:
            proto = G[result['path'][0]][result['path'][1]].get('protocol','?')
            print(f"    ↳ Protocol on chosen edge: {proto} | Latency: {G[result['path'][0]][result['path'][1]].get('latency_cycles','?'):.0f} cycles")
        results.append(result)
        time.sleep(0.01)
    switched = sum(1 for r in results if r.get('congested', False))
    print(f"\n  Summary: {switched}/{n_packets} packets hit congestion")
    return results


def demo_level5(G, src, dst, n_packets=4):
    print("\n" + "─"*62)
    print("  LEVEL 5 — Q-Learning (trained on dataset edge rewards)")
    print("─"*62)
    print("  Training Q-Learning agent… ", end="", flush=True)
    router = QLearningRouter(G, episodes=700, alpha=0.15, gamma=0.9)
    router.train(goal=dst)
    print("Done.")
    results = []
    for pid in range(n_packets):
        G = add_dynamic_traffic(G, intensity=3)
        router.update_graph(G)
        router.train(goal=dst)
        result = router.route_packet(src, dst, packet_id=pid+20)
        log_routing_decision(result)
        if result.get('predicted_risky_edges'):
            print(f"  ⚠ Pkt#{pid+20} — Risky edges: {result['predicted_risky_edges']}")
        results.append(result)
    print(f"\n  Q-Table: {router.get_q_table_summary()}")
    return results


def demo_traffic_model():
    print("\n" + "─"*62)
    print("  TRAFFIC MODEL — Trained on real protocol dataset")
    print("─"*62)
    tm = TrafficModel()
    tm.train()
    stats = tm.get_model_stats()
    print(f"  Model stats: MAE={stats['regression_mae']}  Classifier Acc={stats['classifier_accuracy']}")
    print(f"  Data source: {stats['data_source']}")
    print()
    # Test with real protocol parameter combos
    test_cases = [
        (451.0, 0.03, 0.0, 8, "UART"),
        (761.0, 0.01, 0.0, 16, "SPI"),
        (496.0, 0.009, 0.5, 8, "I2C"),
        (1200.0, 0.5, 1.0, 32, "UART"),
    ]
    for lat, noise, ferr, payload, proto in test_cases:
        pred = tm.predict_load(lat, noise, ferr, payload, proto)
        will_c = tm.will_congest(lat, noise, ferr, payload, proto)
        print(f"  {proto:<5} | lat={lat:.0f}  noise={noise:.3f}  ferr={ferr:.1f}"
              f"  → load={pred:.1f}%  congest={'YES' if will_c else 'no'}")


def main():
    print(BANNER)
    random.seed(42)
    NUM_NODES = 12
    G = create_graph(num_nodes=NUM_NODES, edge_prob=0.45)
    G = add_dynamic_traffic(G, intensity=3)

    stats = get_graph_stats(G)
    log_network_state(stats)
    print(f"\n  Protocol distribution on edges: {stats.get('protocol_distribution', {})}")
    print(f"  Avg latency across graph: {stats.get('avg_latency_cycles', 0):.1f} cycles")

    nodes = list(G.nodes())
    src, dst = nodes[0], nodes[-1]
    print(f"\n  Network: {NUM_NODES} nodes | Source={src} | Destination={dst}")

    demo_dataset()
    demo_level3(G, src, dst, k=5)
    demo_level4(G, src, dst, n_packets=6)
    demo_level5(G, src, dst, n_packets=4)
    demo_traffic_model()

    print("\n" + "═"*62)
    print("  All levels demonstrated. Log saved to logs/routing.log")
    print("═"*62)


if __name__ == "__main__":
    main()
