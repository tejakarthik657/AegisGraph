"""
AegisGraph — Hybrid GNN Model.

Combines the Temporal Graph Network (TGN) with the
Heterogeneous GNN to leverage both temporal and
entity-type-aware features for fraud detection.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HybridFraudDetector(nn.Module):
    """
    Hybrid model combining TGN temporal embeddings
    with heterogeneous GNN structural embeddings.
    """

    def __init__(
        self,
        tgn_embedding_dim: int = 64,
        hgnn_embedding_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        combined_dim = tgn_embedding_dim + hgnn_embedding_dim

        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(
        self,
        tgn_embeddings: torch.Tensor,
        hgnn_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass combining both embedding types.

        Parameters
        ----------
        tgn_embeddings:
            Temporal embeddings from TGN [batch, tgn_dim].

        hgnn_embeddings:
            Structural embeddings from HGNN [batch, hgnn_dim].

        Returns
        -------
        Logits [batch, output_dim].
        """

        combined = torch.cat(
            [tgn_embeddings, hgnn_embeddings],
            dim=-1,
        )

        return self.fusion_layer(combined)

    def predict_proba(
        self,
        tgn_embeddings: torch.Tensor,
        hgnn_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Return fraud probability."""

        logits = self.forward(
            tgn_embeddings, hgnn_embeddings
        )

        return torch.softmax(logits, dim=-1)[:, 1]
