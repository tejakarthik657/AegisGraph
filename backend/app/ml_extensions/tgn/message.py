import torch
import torch.nn as nn


class MessageEncoder(nn.Module):
    """
    Encodes a transaction event together with the current
    sender/receiver memories and temporal information.

    Input:
        sender_memory
        receiver_memory
        transaction_features
        time_embedding

    Output:
        Event message representation.
    """

    def __init__(
        self,
        memory_dim: int = 64,
        transaction_dim: int = 16,
        time_dim: int = 64,
        message_dim: int = 64,
    ):
        super().__init__()

        input_dim = (
            memory_dim
            + memory_dim
            + transaction_dim
            + time_dim
        )

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        sender_memory: torch.Tensor,
        receiver_memory: torch.Tensor,
        transaction_features: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:

        # Make sure one-dimensional inputs become
        # [batch_size, feature_dim].
        if sender_memory.dim() == 1:
            sender_memory = sender_memory.unsqueeze(0)

        if receiver_memory.dim() == 1:
            receiver_memory = receiver_memory.unsqueeze(0)

        if transaction_features.dim() == 1:
            transaction_features = transaction_features.unsqueeze(0)

        if time_embedding.dim() == 1:
            time_embedding = time_embedding.unsqueeze(0)

        event_input = torch.cat(
            [
                sender_memory,
                receiver_memory,
                transaction_features,
                time_embedding,
            ],
            dim=-1,
        )

        return self.encoder(event_input)