import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from basissharing import BSConfig, ModuleSharingConfig


@pytest.fixture
def samples():
    torch.manual_seed(0)
    # Shape: (batch=1, seq=10, hidden=64) — matches MinimalModel input
    return [torch.randn(1, 10, 64) for _ in range(8)]


@pytest.fixture
def batches():
    torch.manual_seed(0)
    # Generate 8 batches of shape (1, 10, 64)
    return [torch.randn(4, 10, 64) for _ in range(8)]


@pytest.fixture
def model():
    return MinimalModel(num_layers=4, hidden=64, ffn_dim=128)


@pytest.fixture
def bs_config():
    return BSConfig(
        model_id="minimal",
        module_cfgs=[
            ModuleSharingConfig("q", group_size=2, compression_ratio=0.5),
            ModuleSharingConfig("up", group_size=2, compression_ratio=0.5),
        ],
    )


class MinimalAttention(nn.Module):
    """Multi-head attention with 2 heads."""

    def __init__(self, hidden: int = 64, num_heads: int = 2):
        super().__init__()
        assert hidden % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.scale = self.head_dim**-0.5

        self.q = nn.Linear(hidden, hidden, bias=False)  # W_Q
        self.k = nn.Linear(hidden, hidden, bias=False)  # W_K
        self.v = nn.Linear(hidden, hidden, bias=False)  # W_V
        self.o = nn.Linear(hidden, hidden, bias=False)  # W_O

    def forward(self, x):
        B, S, H = x.shape
        # Each head must attend over the full hidden dimension, so we project to (B, S, num_heads, head_dim)
        # and then transpose to (B, num_heads, S, head_dim)
        Q = self.q(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.softmax(
            Q @ K.transpose(-2, -1) * self.scale, dim=-1
        )  # (B, num_heads, S, head_dim) @ (B, num_heads, head_dim, S) -> (B, num_heads, S, S)
        out = (
            (scores @ V).transpose(1, 2).reshape(B, S, H)
        )  # (B, num_heads, S, S) @ (B, num_heads, S, head_dim) -> (B, num_heads, S, head_dim) -> (B, S, num_heads * head_dim) = (B, S, H)
        return self.o(out)


class MinimalFFN(nn.Module):
    """Gated FFN matching LLaMA's W_up, W_gate, W_down structure."""

    def __init__(self, hidden: int = 64, ffn_dim: int = 128):
        super().__init__()
        self.up = nn.Linear(hidden, ffn_dim, bias=False)  # W_up
        self.gate = nn.Linear(hidden, ffn_dim, bias=False)  # W_gate
        self.down = nn.Linear(ffn_dim, hidden, bias=False)  # W_down

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MinimalBlock(nn.Module):
    """Transformer block with residuals and layer norms."""

    def __init__(self, hidden: int = 64, ffn_dim: int = 128):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = MinimalAttention(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = MinimalFFN(hidden, ffn_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # residual around attention
        x = x + self.ffn(self.norm2(x))  # residual around FFN
        return x


class MinimalModel(nn.Module):
    """
    Two transformer blocks. Each block contains all matrix types from the paper:
      W_Q, W_K, W_V, W_O        (attention)
      W_up, W_gate               (FFN, eligible for basis sharing)
      W_down                     (FFN, NOT eligible per paper section 3.2)
    """

    def __init__(self, num_layers: int = 4, hidden: int = 64, ffn_dim: int = 128):
        super().__init__()
        torch.manual_seed(42)
        self.blocks = nn.ModuleList(
            [MinimalBlock(hidden, ffn_dim) for _ in range(num_layers)]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
