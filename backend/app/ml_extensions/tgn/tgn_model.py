import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv

from app.ml_extensions.tgn.message import MessageEncoder
from app.ml_extensions.tgn.memory import TemporalMemory
from app.ml_extensions.tgn.time_encoding import TimeEncoder

class TemporalMemoryTGN(nn.Module):
    """
    Event-driven temporal graph model for transaction-level
    fraud prediction.

    Predicts each transaction using existing account memories,
    then updates sender and receiver memories.

    Temporal rule:
        predict(event_t)
        update_memory(event_t)
    """

    def __init__(
        self,
        input_dim,
        num_nodes,
        hidden_dim=64,
        message_dim=64,
        heads=2,
        dropout=0.2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.num_nodes = num_nodes

        # Transaction feature projection
        self.input_projection = nn.Linear(
            input_dim,
            hidden_dim,
        )

        # Temporal encoding
        self.time_encoder = TimeEncoder(
            output_dim=hidden_dim,
        )

        # Persistent account memory
        self.memory = TemporalMemory(
            num_nodes=num_nodes,
            memory_dim=hidden_dim,
            message_dim=message_dim,
        )

        # Event message encoder
        self.message_encoder = MessageEncoder(
            memory_dim=hidden_dim,
            transaction_dim=input_dim,
            time_dim=hidden_dim,
            message_dim=message_dim,
        )

        # Retained from the existing architecture.
        self.gat1 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=heads,
            concat=False,
            dropout=dropout,
        )

        self.gat2 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=heads,
            concat=False,
            dropout=dropout,
        )

        # Combines sender memory, receiver memory and message.
        self.transaction_projection = nn.Linear(
            hidden_dim + hidden_dim + message_dim,
            hidden_dim,
        )

        # Transaction-level fraud classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    # ---------------------------------------------------------
    # Memory access
    # ---------------------------------------------------------

    def get_account_memory(self, node_ids):
        return self.memory.get_memory(node_ids)

    def get_last_update(self, node_ids):
        return self.memory.get_last_update(node_ids)

    # ---------------------------------------------------------
    # Event encoding
    # ---------------------------------------------------------

    def encode_event(
        self,
        sender_ids,
        receiver_ids,
        transaction_features,
        delta_t,
    ):
        """
        Encode an event using account memories from BEFORE
        the event has been committed to memory.
        """

        sender_memory = self.memory.get_memory(sender_ids)
        receiver_memory = self.memory.get_memory(receiver_ids)

        time_embedding = self.time_encoder(delta_t)

        message = self.message_encoder(
            sender_memory=sender_memory,
            receiver_memory=receiver_memory,
            transaction_features=transaction_features,
            time_embedding=time_embedding,
        )

        projected_transaction = self.input_projection(
            transaction_features
        )

        return (
            sender_memory,
            receiver_memory,
            projected_transaction,
            message,
        )

    # ---------------------------------------------------------
    # Event prediction
    # ---------------------------------------------------------

    def predict_event(
        self,
        sender_ids,
        receiver_ids,
        transaction_features,
        delta_t,
    ):
        """
        Return fraud logits for the current transaction.

        Does not commit the current event to account memory.
        """

        (
            sender_memory,
            receiver_memory,
            _projected_transaction,
            message,
        ) = self.encode_event(
            sender_ids=sender_ids,
            receiver_ids=receiver_ids,
            transaction_features=transaction_features,
            delta_t=delta_t,
        )

        transaction_representation = torch.cat(
            [
                sender_memory,
                receiver_memory,
                message,
            ],
            dim=-1,
        )

        transaction_representation = self.transaction_projection(
            transaction_representation
        )

        transaction_representation = F.relu(
            transaction_representation
        )

        logits = self.classifier(
            transaction_representation
        )

        return logits.squeeze(-1)

    # ---------------------------------------------------------
    # Event memory update
    # ---------------------------------------------------------

    def update_event_memory(
        self,
        sender_ids,
        receiver_ids,
        transaction_features,
        delta_t,
        timestamps,
        track_grad=False,
    ):
        """
        Update account memories AFTER the event has been predicted.

        track_grad=True:
            Preserve the computation graph across events within
            a TBPTT window.

        track_grad=False:
            Commit detached memory for evaluation/inference.
        """

        # Encode the event using the pre-update memories.
        _, _, _, message = self.encode_event(
            sender_ids=sender_ids,
            receiver_ids=receiver_ids,
            transaction_features=transaction_features,
            delta_t=delta_t,
        )

        # Compute both next states before committing either one.
        # This ensures both use the same pre-event memories.
        sender_new_memory = self.memory.compute_updated_memory(
            node_ids=sender_ids,
            messages=message,
        )

        is_self_transfer = torch.equal(
            sender_ids,
            receiver_ids,
        )

        receiver_new_memory = None

        if not is_self_transfer:
            receiver_new_memory = (
                self.memory.compute_updated_memory(
                    node_ids=receiver_ids,
                    messages=message,
                )
            )

        # Commit sender state.
        if track_grad:
            self.memory.commit_memory_with_grad(
                node_ids=sender_ids,
                new_memory=sender_new_memory,
                timestamps=timestamps,
            )
        else:
            self.memory.commit_memory(
                node_ids=sender_ids,
                new_memory=sender_new_memory,
                timestamps=timestamps,
            )

        # Commit receiver state only if it is a different account.
        if not is_self_transfer:
            if track_grad:
                self.memory.commit_memory_with_grad(
                    node_ids=receiver_ids,
                    new_memory=receiver_new_memory,
                    timestamps=timestamps,
                )
            else:
                self.memory.commit_memory(
                    node_ids=receiver_ids,
                    new_memory=receiver_new_memory,
                    timestamps=timestamps,
                )

    # ---------------------------------------------------------
    # Reset memory
    # ---------------------------------------------------------

    @torch.no_grad()
    def reset_memory(self):
        """Reset all account memories and timestamps."""
        self.memory.reset_memory()
