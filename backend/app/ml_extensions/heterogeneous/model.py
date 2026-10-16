
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv


class HeterogeneousFraudGNN(nn.Module):
    """
    Heterogeneous GNN for transaction fraud classification.

    Node types:
        - account
        - transaction

    Relations:
        - account sends transaction
        - transaction sent_by account
        - transaction received_by account
        - account receives transaction

    The model produces one fraud logit per transaction node.
    Apply sigmoid to the logits to obtain fraud probabilities.
    """

    def __init__(
        self,
        account_input_dim: int,
        transaction_input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()

        if account_input_dim < 1:
            raise ValueError("account_input_dim must be positive.")

        if transaction_input_dim < 1:
            raise ValueError("transaction_input_dim must be positive.")

        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2.")

        self.dropout = dropout

        self.account_encoder = nn.Linear(
            account_input_dim,
            hidden_dim,
        )

        self.transaction_encoder = nn.Linear(
            transaction_input_dim,
            hidden_dim,
        )

        relations = [
            ("account", "sends", "transaction"),
            ("transaction", "sent_by", "account"),
            ("transaction", "received_by", "account"),
            ("account", "receives", "transaction"),
        ]

        self.conv1 = HeteroConv(
            {
                relation: SAGEConv(
                    (hidden_dim, hidden_dim),
                    hidden_dim,
                )
                for relation in relations
            },
            aggr="sum",
        )

        self.conv2 = HeteroConv(
            {
                relation: SAGEConv(
                    (hidden_dim, hidden_dim),
                    hidden_dim,
                )
                for relation in relations
            },
            aggr="sum",
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, graph: HeteroData) -> torch.Tensor:
        """
        Return one raw fraud logit per transaction node.

        Expected graph node feature keys:
            graph["account"].x
            graph["transaction"].x

        Expected graph edge types:
            The four relation types defined in __init__.
        """

        x_dict = {
            "account": F.relu(
                self.account_encoder(
                    graph["account"].x
                )
            ),
            "transaction": F.relu(
                self.transaction_encoder(
                    graph["transaction"].x
                )
            ),
        }

        x_dict = self.conv1(
            x_dict,
            graph.edge_index_dict,
        )

        x_dict = {
            node_type: F.relu(features)
            for node_type, features in x_dict.items()
        }

        x_dict = {
            node_type: F.dropout(
                features,
                p=self.dropout,
                training=self.training,
            )
            for node_type, features in x_dict.items()
        }

        x_dict = self.conv2(
            x_dict,
            graph.edge_index_dict,
        )

        transaction_embeddings = F.relu(
            x_dict["transaction"]
        )

        logits = self.classifier(
            transaction_embeddings
        )

        return logits.squeeze(-1)
