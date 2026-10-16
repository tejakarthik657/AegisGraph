"""
AegisGraph — Memory Manager for Realtime TGN.

Manages the temporal memory state for real-time
TGN inference, handling memory updates as new
transactions arrive.
"""

from __future__ import annotations

import numpy as np
from collections import OrderedDict


class MemoryManager:
    """
    Manages temporal memory for real-time TGN inference.

    Implements an LRU cache for node memory states.
    """

    def __init__(
        self,
        memory_dim: int = 64,
        max_nodes: int = 100000,
    ):
        self.memory_dim = memory_dim
        self.max_nodes = max_nodes
        self._memory: OrderedDict[str, np.ndarray] = OrderedDict()

    def get_memory(self, node_id: str) -> np.ndarray:
        """Retrieve memory for a node (zero-init if unseen)."""

        if node_id in self._memory:
            self._memory.move_to_end(node_id)
            return self._memory[node_id]

        return np.zeros(self.memory_dim, dtype=np.float32)

    def update_memory(
        self,
        node_id: str,
        new_memory: np.ndarray,
    ) -> None:
        """Update memory for a node."""

        self._memory[node_id] = new_memory
        self._memory.move_to_end(node_id)

        # Evict oldest if over capacity
        while len(self._memory) > self.max_nodes:
            self._memory.popitem(last=False)

    def reset(self) -> None:
        """Clear all memory states."""
        self._memory.clear()

    @property
    def node_count(self) -> int:
        """Number of nodes with stored memory."""
        return len(self._memory)

    def has_memory(self, node_id: str) -> bool:
        """Check if a node has stored memory."""
        return node_id in self._memory
