"""projector.py — The trainable connector between Surya's tokens and the language model."""

from torch import nn


class Projector(nn.Module):
    """Two-layer MLP with GELU, copied unchanged from notebook 02 (cell 23).

    Maps each of Surya's 1280-dimensional tokens to the language model's 4096-dimensional embedding
    space. About 22 M parameters; it is the only part of the model that is trained.
    """

    def __init__(self, in_dim: int = 1280, out_dim: int = 4096) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
