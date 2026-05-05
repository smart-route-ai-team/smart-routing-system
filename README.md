# 🧠 AI-Based Smart NoC Routing System v2.0

## Overview

An intelligent, multi-level packet routing system for **Network-on-Chip (NoC)** architectures.
**v2.0 fully integrates the `augmented_protocol_dataset.csv`** (30,000 real UART/I2C/SPI samples)
to drive edge weights, congestion detection, and ML model training — replacing all synthetic data.

---

## 📊 Dataset Integration — `augmented_protocol_dataset.csv`

| Column | Used For |
|---|---|
| `latency_cycles` | Edge weight + congestion threshold |
| `throughput_bps` | Load % calculation per edge |
| `noise` | Effective cost penalty (get_edge_effective_weight) |
| `frame_err` | Congestion flag + ML feature |
| `payload_bits` | TrafficModel feature |
| `protocol` | Edge label (UART/I2C/SPI), affects routing display |

### Where the dataset plugs in

```
augmented_protocol_dataset.csv
        │
        ├─► utils/dataset_loader.py  (ProtocolDataset class)
        │         │
        │         ├─► core/graph.py           annotate_graph() / refresh_edge_traffic()
        │         │        → every edge gets real latency, throughput, noise, frame_err, protocol
        │         │
        │         ├─► utils/traffic_model.py  TrafficModel.train()
        │         │        → regression (load %) + classifier (congested Y/N) on 30k rows
        │         │
        │         └─► core/simulator.py       path_avg_latency()
        │                  → reports dataset latency through chosen path
        │
        └─► main.py   demo_dataset()  →  prints live dataset summary
```

---

## 📁 Folder Structure

```
smart_routing_system/
│
├── core/
│   ├── graph.py          # Dataset-annotated graph creation + traffic refresh
│   └── simulator.py      # Cost, latency, congestion simulation
│
├── level3/
│   └── basic_routing.py  # K-shortest paths; shows protocol on each edge
│
├── level4/
│   └── adaptive_routing.py  # Congestion-aware routing + PathMonitor
│
├── level5/
│   └── qlearning_router.py  # Q-Learning with dataset-weighted rewards
│
├── utils/
│   ├── dataset_loader.py    # ← NEW: ProtocolDataset, annotate_graph, get_training_data
│   ├── traffic_model.py     # Upgraded: trains on real 30k-row dataset
│   └── logger.py            # Structured JSON routing log
│
├── data/
│   └── augmented_protocol_dataset.csv
│
├── tests/
│   └── test_routing.py
│
├── main.py      # Orchestrator with demo_dataset() section
└── README.md
```

---

## 🚀 Levels of Intelligence

### 🔹 Level 3 – Basic Intelligent Routing
- K-shortest paths (Yen's), Dijkstra, A*
- Edge costs derived from **real dataset latency** (not random)
- Path output shows **protocol type** (UART/I2C/SPI) per edge

### 🔹 Level 4 – Adaptive Routing
- Real-time congestion detection using **dataset thresholds** (latency_cycles > P75)
- Effective cost = weight × load_factor + **noise_penalty** + **frame_err_penalty**
- Prints actual protocol and latency for each chosen path

### 🔹 Level 5 – Autonomous Routing (Q-Learning)
- Rewards shaped by **real edge noise and frame_err**
- CongestionPredictor trained on live dataset samples per edge
- TrafficModel trained on all 30,000 rows → classifier accuracy ~100%

---

## ⚙️ Setup & Run

```bash
pip install networkx scikit-learn numpy pandas

# Run full demo
cd smart_routing_system
python main.py

# Run tests
python -m pytest tests/ -v
```

---

## 📊 Key Design Decisions

| Concern | v1 (old) | v2 (dataset-driven) |
|---|---|---|
| Edge weights | random.randint(1,10) | latency_cycles from dataset |
| Load % | random.randint(0,100) | derived from throughput_bps |
| Congestion flag | load > 75% | latency > P75 OR frame_err > 0.5 |
| Effective cost | base × load_factor | + noise penalty + frame_err penalty |
| ML training data | synthetic Gaussian | real 30,000 protocol samples |
| Protocol awareness | none | UART/I2C/SPI per edge |
