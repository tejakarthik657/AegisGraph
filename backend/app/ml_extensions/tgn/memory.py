import torch
import torch.nn as nn


class TemporalMemory(nn.Module):
    """
    Maintains a temporal memory vector for every account.

    Each account has:

    - memory vector
    - timestamp of its latest observed event

    The memory is updated after a transaction has been observed.
    """

    def __init__(
        self,
        num_nodes: int,
        memory_dim: int = 64,
        message_dim: int = 64,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.message_dim = message_dim

        # Persistent account memory.
        #
        # These are buffers rather than trainable parameters because
        # they represent the current temporal state of accounts.
        self.register_buffer(
            "memory",
            torch.zeros(
                num_nodes,
                memory_dim,
                dtype=torch.float32,
            ),
        )

        # Latest event timestamp observed for every account.
        self.register_buffer(
            "last_update",
            torch.zeros(
                num_nodes,
                dtype=torch.float32,
            ),
        )

        # Learned memory updater.
        #
        # The updater receives the event message and the previous
        # account memory and produces the next account memory.
        self.memory_updater = nn.GRUCell(
            input_size=message_dim,
            hidden_size=memory_dim,
        )

    # ---------------------------------------------------------
    # Read memory
    # ---------------------------------------------------------

    def get_memory(
        self,
        node_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return current memory vectors for the requested accounts.
        """

        return self.memory[node_ids]

    # ---------------------------------------------------------
    # Read latest update time
    # ---------------------------------------------------------

    def get_last_update(
        self,
        node_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the latest observed timestamp for the requested
        accounts.
        """

        return self.last_update[node_ids]

    # ---------------------------------------------------------
    # Compute next memory
    # ---------------------------------------------------------

    def compute_updated_memory(
        self,
        node_ids: torch.Tensor,
        messages: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the next memory state without modifying the
        persistent memory buffer.

        This separation is important because the caller may want
        to compute the next state first and commit it only after
        the current event has been processed.
        """

        old_memory = self.memory[node_ids]

        new_memory = self.memory_updater(
            messages,
            old_memory,
        )

        return new_memory

    # ---------------------------------------------------------
    # Commit memory
    # ---------------------------------------------------------
    
    @torch.no_grad()
    def commit_memory(
        self,
        node_ids: torch.Tensor,
        new_memory: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> None:
        """Commit detached memory for evaluation/inference."""
        updated_memory = self.memory.index_copy(
            0,
            node_ids,
            new_memory.detach(),
        )
        self.memory = updated_memory

        updated_times = self.last_update.index_copy(
            0,
            node_ids,
            timestamps.detach(),
        )
        self.last_update = updated_times

    def commit_memory_with_grad(
        self,
        node_ids: torch.Tensor,
        new_memory: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> None:
        """Commit memory while retaining gradients within a training window."""
        self.memory = self.memory.index_copy(
            0,
            node_ids,
            new_memory,
        )
        with torch.no_grad():
            self.last_update.index_copy_(
                0,
                node_ids,
                timestamps.detach(),
            )

    def detach_memory(self) -> None:
        """End the gradient history at a TBPTT window boundary."""
        self.memory = self.memory.detach()


    # ---------------------------------------------------------
    # Reset
    # ---------------------------------------------------------

    @torch.no_grad()
    def reset_memory(self) -> None:
        """
        Reset all account memories and timestamps.
        """

        self.memory.zero_()
        self.last_update.zero_()