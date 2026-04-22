"""
utils/dataset_loader.py  [UPGRADED]
--------------------------------------
Loads and processes augmented_protocol_dataset.csv.

KEY UPGRADES:
  1. Protocol fitness scoring — defines which protocol performs best
     for which data type (payload size, noise level, latency budget).
  2. Retransmit tracking — since the dataset has no retransmit column,
     we SIMULATE it: every dropped packet (frame_err > 0) is flagged
     and a retransmit attempt is recorded with outcome tracking.
  3. Drop rate is exposed per-protocol so the router can penalise
     unreliable protocols on a given edge type.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
import networkx as nx
import random

DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'augmented_protocol_dataset.csv')

# ── Protocol baseline stats (computed from dataset) ─────────────────────────
PROTOCOL_STATS = {
    "UART": {
        "avg_latency":    465.2,
        "avg_throughput": 41611.0,
        "avg_noise":      0.0351,
        "drop_rate":      0.0351,
        "best_for":       "small_payload_low_noise",
        "description":    "Best for small payloads (<=8 b) on low-noise links. "
                          "Highest drop rate — retransmit logic is critical.",
    },
    "I2C":  {
        "avg_latency":    496.6,
        "avg_throughput": 46783.8,
        "avg_noise":      0.0090,
        "drop_rate":      0.0092,
        "best_for":       "tiny_payload_multi_device",
        "description":    "Best throughput per bit for tiny payloads (<=8 b). "
                          "Clock-stretch adds latency but drops are very rare.",
    },
    "SPI":  {
        "avg_latency":    761.1,
        "avg_throughput": 41545.8,
        "avg_noise":      0.0084,
        "drop_rate":      0.0069,
        "best_for":       "large_payload_full_duplex",
        "description":    "Best for large payloads (>8 b) and full-duplex routes. "
                          "Lowest drop rate but highest latency on most rows.",
    },
}

LATENCY_P75    = 451.0
LATENCY_P90    = 990.6
MAX_LATENCY    = 1651.0
MAX_THROUGHPUT = 72700.84


# ── Protocol fitness ─────────────────────────────────────────────────────────

def select_best_protocol(payload_bits: float,
                          noise: float,
                          latency_budget: float = 500.0,
                          full_duplex: bool = False) -> str:
    """
    Return the best protocol for the given edge characteristics.

    Decision rules derived from dataset analysis:

    | Condition                          | Winner |
    |------------------------------------|--------|
    | payload <= 8 b  AND  noise < 0.02  | I2C    |
    | payload > 8 b   (any duplex)       | SPI    |
    | noise >= 0.05                      | I2C/SPI|
    | latency_budget < 460 cycles        | UART   |
    | default                            | I2C    |
    """
    if noise >= 0.05:
        return "SPI" if payload_bits > 8 else "I2C"
    if payload_bits > 8:
        return "SPI"
    if latency_budget < 460:
        return "UART"
    return "I2C"


def protocol_fitness_score(protocol: str,
                            payload_bits: float,
                            noise: float,
                            latency_cycles: float) -> float:
    """Return a fitness score [0..1]; higher = more appropriate for this edge."""
    best  = select_best_protocol(payload_bits, noise, latency_cycles)
    stats = PROTOCOL_STATS[protocol]
    tp_score  = min(stats["avg_throughput"] / MAX_THROUGHPUT, 1.0)
    lat_score = 1.0 - min(stats["avg_latency"] / MAX_LATENCY, 1.0)
    rel_score = 1.0 - stats["drop_rate"]
    match_bonus = 0.2 if protocol == best else 0.0
    return round(min(1.0, 0.3*tp_score + 0.3*lat_score + 0.4*rel_score + match_bonus), 4)


# ── Retransmit simulation ────────────────────────────────────────────────────

class RetransmitTracker:
    """
    Tracks per-edge packet drops and ARQ retransmit attempts.

    The dataset has NO retransmit column — we derive it:
      • A packet is DROPPED  if frame_err > 0.
      • A retransmit is auto-attempted (ARQ-style, up to MAX_RETRIES).
      • Retransmit success probability = (1 - protocol_drop_rate).
    """
    MAX_RETRIES = 3

    def __init__(self):
        self._stats: Dict[tuple, dict] = {}

    def _key(self, u, v):
        return (min(u, v), max(u, v))

    def record_transmission(self, u, v, frame_err: float, protocol: str) -> dict:
        k = self._key(u, v)
        if k not in self._stats:
            self._stats[k] = {"total": 0, "dropped": 0, "retried": 0,
                               "recovered": 0, "perm_failed": 0}
        s = self._stats[k]
        s["total"] += 1

        if frame_err == 0:
            return {"dropped": False, "retried": False,
                    "recovered": False, "attempts": 1, "final_ok": True}

        s["dropped"] += 1
        drop_rate = PROTOCOL_STATS.get(protocol, {}).get("drop_rate", 0.05)
        recovered = False
        attempts  = 1
        for _ in range(self.MAX_RETRIES):
            attempts += 1
            s["retried"] += 1
            if random.random() > drop_rate:
                recovered = True
                s["recovered"] += 1
                break
        if not recovered:
            s["perm_failed"] += 1

        return {"dropped": True, "retried": True,
                "recovered": recovered, "attempts": attempts,
                "final_ok": recovered}

    def get_edge_stats(self, u, v) -> dict:
        k = self._key(u, v)
        s = self._stats.get(k, {"total":0,"dropped":0,"retried":0,
                                  "recovered":0,"perm_failed":0})
        total = max(s["total"], 1)
        return {
            **s,
            "drop_rate_pct":       round(s["dropped"]    / total * 100, 2),
            "retransmit_rate_pct": round(s["retried"]    / total * 100, 2),
            "recovery_rate_pct":   round(s["recovered"]  / max(s["dropped"],1)*100, 2),
            "perm_fail_rate_pct":  round(s["perm_failed"]/ total * 100, 2),
        }

    def get_all_stats(self) -> Dict[str, dict]:
        return {f"({u},{v})": self.get_edge_stats(u, v) for (u,v) in self._stats}

    def summary(self) -> dict:
        all_s = list(self._stats.values())
        if not all_s:
            return {}
        total   = sum(s["total"]       for s in all_s)
        dropped = sum(s["dropped"]     for s in all_s)
        retried = sum(s["retried"]     for s in all_s)
        recov   = sum(s["recovered"]   for s in all_s)
        failed  = sum(s["perm_failed"] for s in all_s)
        return {
            "total_packets":     total,
            "total_dropped":     dropped,
            "total_retried":     retried,
            "total_recovered":   recov,
            "total_perm_failed": failed,
            "overall_drop_pct":  round(dropped / max(total,1) * 100, 2),
            "retransmit_pct":    round(retried  / max(total,1) * 100, 2),
            "recovery_pct":      round(recov    / max(dropped,1)*100, 2),
        }


# ── ProtocolDataset ──────────────────────────────────────────────────────────

class ProtocolDataset:
    def __init__(self, path: str = DATASET_PATH):
        self.path   = path
        self._df: Optional[pd.DataFrame] = None
        self._by_protocol: Dict[str, pd.DataFrame] = {}
        self.retransmit_tracker = RetransmitTracker()

    def load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
        self._df = pd.read_csv(self.path)
        for proto in ["UART", "I2C", "SPI"]:
            self._by_protocol[proto] = (
                self._df[self._df["protocol"] == proto].reset_index(drop=True)
            )
        return self._df

    @property
    def df(self) -> pd.DataFrame:
        return self.load()

    def sample_edge_metrics(self, protocol: Optional[str] = None,
                             u=0, v=1) -> dict:
        df = self.load()
        if protocol and protocol in self._by_protocol:
            row = self._by_protocol[protocol].sample(1).iloc[0]
        else:
            row = df.sample(1).iloc[0]

        latency    = float(row["latency_cycles"])
        throughput = float(row["throughput_bps"])
        noise      = float(row["noise"])
        frame_err  = float(row["frame_err"])
        payload    = float(row["payload_bits"])
        proto_str  = str(row["protocol"])

        weight    = max(1, int(round(latency / MAX_LATENCY * 10)))
        load_pct  = max(0, min(100, int((1 - throughput / MAX_THROUGHPUT) * 100)))
        congested = latency > LATENCY_P75 or frame_err > 0.5

        fitness    = protocol_fitness_score(proto_str, payload, noise, latency)
        best_proto = select_best_protocol(payload, noise, latency)
        retx_info  = self.retransmit_tracker.record_transmission(
            u, v, frame_err, proto_str)

        return {
            "weight":           weight,
            "load":             load_pct,
            "congested":        congested,
            "latency_cycles":   latency,
            "throughput_bps":   throughput,
            "noise":            noise,
            "frame_err":        frame_err,
            "payload_bits":     payload,
            "protocol":         proto_str,
            "protocol_fitness": fitness,
            "best_protocol":    best_proto,
            "retransmit_info":  retx_info,
        }

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        df = self.load()
        payload_norm = df["payload_bits"].fillna(8) / 32.0
        X = np.column_stack([
            df["noise"].fillna(0).values,
            df["frame_err"].fillna(0).values,
            payload_norm.values,
            (df["protocol"] == "UART").astype(float).values,
            (df["protocol"] == "I2C").astype(float).values,
            (df["protocol"] == "SPI").astype(float).values,
        ])
        lat = df["latency_cycles"]
        y = ((lat - lat.min()) / (lat.max() - lat.min() + 1e-9) * 100).values
        return X.astype(np.float32), y.astype(np.float32)

    def get_congestion_labels(self) -> np.ndarray:
        df = self.load()
        return ((df["latency_cycles"] > 500) | (df["frame_err"] > 0)).astype(int).values

    def summary(self) -> dict:
        df = self.load()
        congested_count = int(
            ((df["latency_cycles"] > LATENCY_P75) | (df["frame_err"] > 0.5)).sum()
        )
        return {
            "total_rows":          len(df),
            "protocols":           df["protocol"].value_counts().to_dict(),
            "avg_latency_cycles":  round(df["latency_cycles"].mean(), 2),
            "avg_throughput_bps":  round(df["throughput_bps"].mean(), 2),
            "avg_noise":           round(df["noise"].mean(), 4),
            "avg_frame_err":       round(df["frame_err"].mean(), 4),
            "congested_rows":      congested_count,
            "congestion_rate_pct": round(congested_count / len(df) * 100, 2),
            "protocol_fitness": {
                p: {
                    "best_for":    PROTOCOL_STATS[p]["best_for"],
                    "drop_rate":   PROTOCOL_STATS[p]["drop_rate"],
                    "description": PROTOCOL_STATS[p]["description"],
                }
                for p in ["UART", "I2C", "SPI"]
            },
        }

    # REPLACE annotate_graph with this:

    def annotate_graph(self, G: nx.Graph) -> nx.Graph:
        for u, v, data in G.edges(data=True):
            metrics = self.sample_edge_metrics(u=u, v=v)
            best    = select_best_protocol(
                metrics["payload_bits"], metrics["noise"], metrics["latency_cycles"]
            )
            metrics["best_protocol"]    = best
            metrics["protocol_fitness"] = protocol_fitness_score(
                metrics["protocol"], metrics["payload_bits"],
                metrics["noise"],    metrics["latency_cycles"]
            )
            data.update(metrics)
        return G

    # REPLACE refresh_edge_traffic with this:

    # REPLACE refresh_edge_traffic with this:

    def refresh_edge_traffic(self, G: nx.Graph) -> nx.Graph:
        for u, v, data in G.edges(data=True):
            # Only fix 60% of mismatched edges per refresh click
            current_proto = data.get("protocol")
            best_proto = select_best_protocol(
                data.get("payload_bits", 8.0),
                data.get("noise", 0.03),
                data.get("latency_cycles", 500.0),
            )
            if current_proto != best_proto:
                if random.random() < 0.60:  # fix 60% of mismatches per click
                    data["protocol"]         = best_proto
                    data["best_protocol"]    = best_proto
                    data["protocol_fitness"] = protocol_fitness_score(
                        best_proto,
                        data.get("payload_bits", 8.0),
                        data.get("noise", 0.03),
                        data.get("latency_cycles", 500.0),
                    )
            else:
                # already optimal — just update load/weight
                data["best_protocol"]    = best_proto
                data["protocol_fitness"] = protocol_fitness_score(
                    best_proto,
                    data.get("payload_bits", 8.0),
                    data.get("noise", 0.03),
                    data.get("latency_cycles", 500.0),
                )
        return G

    def get_retransmit_report(self) -> dict:
        return {
            "summary":  self.retransmit_tracker.summary(),
            "per_edge": self.retransmit_tracker.get_all_stats(),
            "protocol_drop_rates": {
                p: PROTOCOL_STATS[p]["drop_rate"] for p in PROTOCOL_STATS
            },
        }

    def get_protocol_recommendation(self, payload_bits: float,
                                     noise: float,
                                     latency_budget: float = 500.0,
                                     full_duplex: bool = False) -> dict:
        best   = select_best_protocol(payload_bits, noise, latency_budget, full_duplex)
        ranked = sorted(
            PROTOCOL_STATS.keys(),
            key=lambda p: protocol_fitness_score(p, payload_bits, noise, latency_budget),
            reverse=True,
        )
        return {
            "recommended": best,
            "ranking":     ranked,
            "scores": {
                p: protocol_fitness_score(p, payload_bits, noise, latency_budget)
                for p in PROTOCOL_STATS
            },
            "rationale": PROTOCOL_STATS[best]["description"],
        }


_dataset_instance: Optional[ProtocolDataset] = None


def get_dataset() -> ProtocolDataset:
    global _dataset_instance
    if _dataset_instance is None:
        _dataset_instance = ProtocolDataset()
        _dataset_instance.load()
    return _dataset_instance
