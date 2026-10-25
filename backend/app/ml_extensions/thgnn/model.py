"""Temporal Heterogeneous GNN for transaction-level fraud classification.

The model uses typed account/transaction message passing and an explicit
continuous-time feature encoder. It is a compact temporal-aware heterogeneous
GNN intended as an experimental addition to AegisGraph, not a claim to reproduce
one particular published THGNN architecture.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv


class TemporalHeterogeneousFraudGNN(nn.Module):
    """Predict a fraud logit for every transaction node in a HeteroData graph."""

    RELATIONS = (
        ("account", "sends", "transaction"),
        ("transaction", "sent_by", "account"),
        ("transaction", "received_by", "account"),
        ("account", "receives", "transaction"),
    )

    def __init__(
        self,
        account_input_dim: int,
        transaction_input_dim: int,
        hidden_dim: int = 64,
        temporal_dim: int = 16,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if min(account_input_dim, transaction_input_dim, temporal_dim) < 1:
            raise ValueError("Input and temporal dimensions must be positive.")
        if hidden_dim < 4:
            raise ValueError("hidden_dim must be at least 4.")

        self.dropout = dropout
        self.account_encoder = nn.Linear(account_input_dim, hidden_dim)
        self.transaction_encoder = nn.Linear(transaction_input_dim, hidden_dim)
        self.temporal_encoder = nn.Sequential(
            nn.Linear(2, temporal_dim),
            nn.ReLU(),
            nn.Linear(temporal_dim, hidden_dim),
        )
        self.conv1 = HeteroConv(
            {rel: SAGEConv((hidden_dim, hidden_dim), hidden_dim) for rel in self.RELATIONS},
            aggr="sum",
        )
        self.conv2 = HeteroConv(
            {rel: SAGEConv((hidden_dim, hidden_dim), hidden_dim) for rel in self.RELATIONS},
            aggr="sum",
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, graph: HeteroData) -> torch.Tensor:
        if "time_features" not in graph["transaction"]:
            raise ValueError(
                "THGNN graph is missing transaction time_features; "
                "build it with build_thgnn_graph()."
            )

        account_x = F.relu(self.account_encoder(graph["account"].x))
        transaction_x = F.relu(self.transaction_encoder(graph["transaction"].x))
        temporal_x = self.temporal_encoder(
            graph["transaction"].time_features.to(transaction_x.dtype)
        )
        x_dict = {"account": account_x, "transaction": transaction_x + temporal_x}

        x_dict = self.conv1(x_dict, graph.edge_index_dict)
        x_dict = {
            node_type: F.dropout(F.relu(features), p=self.dropout, training=self.training)
            for node_type, features in x_dict.items()
        }
        x_dict = self.conv2(x_dict, graph.edge_index_dict)
        transaction_embeddings = F.relu(x_dict["transaction"])
        return self.classifier(transaction_embeddings).squeeze(-1)
