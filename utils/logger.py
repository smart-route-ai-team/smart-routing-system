"""
utils/logger.py
---------------
Structured logging for routing decisions across all levels.
Writes to both console and logs/routing.log.
"""

import json
import os
import time
from datetime import datetime
from typing import Any

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'routing.log')


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def log_routing_decision(result: dict, verbose: bool = True):
    """Log a routing decision result dict."""
    entry = {"timestamp": _timestamp(), **result}
    line = json.dumps(entry)

    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

    if verbose:
        level = result.get('level', '?')
        pid = result.get('packet_id', '?')
        path = result.get('path', [])
        cost = result.get('cost', '?')
        success = result.get('success', False)
        status = "✓" if success else "✗"
        print(f"[L{level}] Pkt#{pid} {status} | Path: {path} | Cost: {cost}")


def log_network_state(stats: dict):
    """Log a network state snapshot."""
    entry = {"timestamp": _timestamp(), "type": "network_state", **stats}
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(entry) + '\n')
    print(f"[NET] Nodes:{stats.get('nodes')} Edges:{stats.get('edges')} "
          f"Congested:{stats.get('congested_edges')} AvgLoad:{stats.get('avg_load_percent')}%")


def read_log(tail: int = 20) -> list:
    """Return the last N log entries."""
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE) as f:
        lines = f.readlines()
    return [json.loads(l) for l in lines[-tail:]]
