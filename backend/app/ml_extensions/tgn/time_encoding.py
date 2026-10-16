import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    """
    Encodes elapsed time into a learned temporal representation.

    Input:
        delta_t: elapsed time since the previous relevant event.

    Output:
        Learned temporal embedding of shape [batch_size, output_dim].
    """

    def __init__(
        self,
        output_dim: int = 64,
    ):
        super().__init__()

        self.output_dim = output_dim

        self.encoder = nn.Sequential(
            nn.Linear(1, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:

        # Make sure input is [batch_size, 1]
        if delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(-1)

        # Numerical stability for potentially large time gaps.
        delta_t = torch.log1p(
            torch.clamp(delta_t, min=0.0)
        )

        return self.encoder(delta_t)