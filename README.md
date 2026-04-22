# 🧠 AI-Based Smart NoC Routing System v2.0

> Multi-level intelligent packet routing for **Network-on-Chip (NoC)** architectures,  
> powered by 30,000 real UART/I2C/SPI samples and a full Protocol Intelligence engine.

---

## 📌 What This System Does

This project simulates an intelligent routing system for NoC architectures across **3 levels of intelligence** — from classical graph algorithms to reinforcement learning — with every routing decision backed by a real protocol dataset.

**Key capabilities:**
- Routes packets using K-Shortest Paths, Adaptive (congestion-aware), and Q-Learning
- Assigns UART / I2C / SPI protocols to edges based on real dataset samples
- Detects protocol mismatches and optimizes assignments via the Protocol Intelligence engine
- Tracks packet drops, retransmits, and recoveries using simulated ARQ
- Visualizes everything through an interactive Streamlit dashboard

---

## 📊 Dataset — `augmented_protocol_dataset.csv`

| Column | Used For |
|---|---|
| `latency_cycles` | Edge weight + congestion threshold |
| `throughput_bps` | Load % calculation per edge |
| `noise` | Effective cost penalty in routing |
| `frame_err` | Congestion flag + drop simulation |
| `payload_bits` | Protocol selection + TrafficModel feature |
| `protocol` | Edge label (UART/I2C/SPI) |

**Dataset stats (30,000 rows):**

| Protocol | Count | Avg Latency | Drop Rate |
|---|---|---|---|
| UART | 10,000 | 465.2 cycles | 3.51% |
| I2C | 10,035 | 496.6 cycles | 0.92% |
| SPI | 9,965 | 761.1 cycles | 0.69% |

---

## 📁 Folder Structure

```
smart_routing_fixed/
│
├── core/
│   ├── graph.py              # Graph creation, annotate_graph, add_dynamic_traffic
│   └── simulator.py          # Cost, latency, drop simulation
│
├── level3/
│   └── basic_routing.py      # K-Shortest Paths (Yen's), Dijkstra, A*
│
├── level4/
│   └── adaptive_routing.py   # Congestion-aware routing + PathMonitor
│
├── level5/
│   └── qlearning_router.py   # Q-Learning with dataset-weighted rewards
│
├── utils/
│   ├── dataset_loader.py     # ProtocolDataset, annotate_graph, Protocol Intelligence
│   ├── traffic_model.py      # TrafficModel trained on 30k dataset rows
│   └── logger.py             # Structured JSON routing log
│
├── data/
│   └── augmented_protocol_dataset.csv
│
├── tests/
│   └── test_routing.py
│
├── app.py       # Streamlit dashboard (7 tabs)
├── main.py      # CLI orchestrator
└── README.md
```

---

## 🚀 Levels of Intelligence

### 🔹 Level 3 – Basic Intelligent Routing
- K-Shortest Paths (Yen's algorithm), Dijkstra, A*
- Edge costs derived from real dataset latency values
- Displays protocol type (UART/I2C/SPI) and fitness score per path

### 🔹 Level 4 – Adaptive Routing
- Congestion detection using dataset thresholds (latency > P75 or frame_err > 0.5)
- Effective cost = base × load_factor + noise_penalty + frame_err_penalty + mismatch_penalty
- PathMonitor tracks attempts, successes, congestion events, and avg latency per route

### 🔹 Level 5 – Autonomous Routing (Q-Learning)
- Reward shaped by real edge noise and frame_err from dataset
- CongestionPredictor trained on live dataset samples per edge
- Q-Table converges after ~500 episodes; finds minimum-cost path autonomously
- Reward Convergence chart shows learning curve stabilization

---

## 📡 Protocol Intelligence Engine

The system includes a full protocol selection engine that determines the optimal protocol for each edge based on its traffic characteristics.

### Protocol Decision Rules

| Condition | Best Protocol | Reason |
|---|---|---|
| Payload ≤ 8 b, noise < 0.02 | I2C | Highest throughput-per-bit, drop rate only 0.92% |
| Payload > 8 b (any duplex) | SPI | Handles large frames, lowest drop rate 0.69% |
| Noise ≥ 0.05 (any payload) | I2C / SPI | UART drop rate jumps to ~75% at high noise |
| Latency budget < 460 cycles | UART | Lowest median latency (451 cycles) |
| Default / balanced | I2C | Best overall throughput per bit, minimal drops |

### Mismatch Detection & Optimization Loop

When a network is generated, edges are assigned protocols from the dataset as-is — reflecting real-world conditions where protocols may not be optimally configured.

- **Protocol Mismatch** = edge is using a protocol that is not the best fit for its current noise, payload, and latency
- **Avg Protocol Fitness** = mean score (0–1) across all edges; 1.0 means every edge is on its optimal protocol
- **Mismatch Penalty** = `(1.0 - fitness) × 2.0` added to routing cost, so mismatched edges are naturally avoided

**Optimization flow:**

```
Generate Network
    → Real protocols assigned from dataset (some suboptimal)
    → Protocol Mismatches: ~10–20, Avg Fitness: ~0.87

Refresh Traffic (click 1)
    → ~60% of mismatched edges reassigned to optimal protocol
    → Protocol Mismatches: ~6–8, Avg Fitness: ~0.92

Refresh Traffic (click 2)
    → Another ~60% of remaining mismatches resolved
    → Protocol Mismatches: ~2–3, Avg Fitness: ~0.96

Refresh Traffic (click 3)
    → All edges optimal
    → Protocol Mismatches: 0, Avg Fitness: ~0.97
```

---

## 🔁 Packet Drop & Retransmit (ARQ Simulation)

The dataset has no retransmit column — the system simulates **Automatic Repeat reQuest (ARQ)**:

- A packet is **dropped** when `frame_err > 0`
- A **retransmit** is automatically triggered (up to 3 attempts)
- Retransmit success probability = `1 - protocol_drop_rate`
- If all 3 attempts fail → **permanently lost**

---

## 🖥️ Streamlit Dashboard — 7 Tabs

| Tab | What It Shows |
|---|---|
| Network View | Interactive graph with colored edges (UART/I2C/SPI), edge status table |
| Basic | K-Shortest path routing with candidate path comparison |
| Adaptive | Congestion-aware routing, cost-per-packet chart, PathMonitor report |
| Q-Learning | RL agent training, reward convergence, Q-Table heatmap |
| Protocol Intel | Protocol recommendation sliders, fitness scores, decision map, mismatch counter |
| Drop & Retransmit | ARQ simulation results, drop rates by protocol, per-edge retransmit details |
| Analytics | Session summary, cost distribution by routing level, traffic predictions |

---

## ⚙️ Setup & Run

```bash
# Install dependencies
pip install networkx scikit-learn numpy pandas streamlit plotly

# Run Streamlit dashboard
streamlit run app.py

# Run CLI demo
python main.py

# Run tests
python -m pytest tests/ -v
```

---

## 📈 Key Design Decisions

| Concern | v1 (old) | v2 (dataset-driven) |
|---|---|---|
| Edge weights | `random.randint(1,10)` | `latency_cycles` from dataset |
| Load % | `random.randint(0,100)` | Derived from `throughput_bps` |
| Congestion flag | `load > 75%` | `latency > P75` OR `frame_err > 0.5` |
| Effective cost | `base × load_factor` | + noise + frame_err + mismatch penalty |
| ML training data | Synthetic Gaussian | Real 30,000 protocol samples |
| Protocol awareness | None | UART/I2C/SPI per edge with fitness scoring |
| Protocol optimization | None | Mismatch detection + multi-step refresh convergence |
| Retransmit tracking | None | Full ARQ simulation with recovery stats |

---

## 🧪 Sample CLI Output

```
[NET] Nodes:12  Edges:32  Congested:11  AvgLoad:55.66%
Protocol distribution: {'SPI': 14, 'UART': 9, 'I2C': 9}

[L3] Pkt#1 ✓ | Path: [0, 10, 11] | Cost: 3
[L4] Pkt#10 ✓ | Path: [0, 11]    | Cost: 1  | Protocol: SPI | Latency: 451 cycles
[L5] Pkt#20 ✓ | Path: [0, 11]    | Cost: 1  (Q-Learning converged)

Q-Table: shape=(12,12)  max_q=98.99  nonzero=58
TrafficModel: MAE=9.858  Classifier Acc=87.22%
```

---

## 👨‍💻 Tech Stack

| Technology | Usage |
|---|---|
| Python 3.10 | Core language |
| NetworkX | Graph creation and pathfinding |
| Pandas / NumPy | Dataset processing |
| Scikit-learn | TrafficModel (regression + classifier) |
| Streamlit | Interactive dashboard |
| Plotly | Charts and network visualization |