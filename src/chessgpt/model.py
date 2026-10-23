"""A small GPT (decoder-only transformer), nanoGPT style.

Several games can be packed into one training row. `pos` gives each token's
position within its own game and `mask` stops attention across games, so every
game is seen exactly as if it were alone in the row.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int  # max tokens per game
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            is_causal=mask is None,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), mask))
        return x + self.drop(self.mlp(self.ln2(x)))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)
        # Scale residual projections down so the residual stream doesn't grow with depth (GPT-2).
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters()) - self.pos_emb.weight.numel()

    def forward(
        self,
        idx: torch.Tensor,
        pos: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """idx, pos: (B, T) token ids and in-game positions. mask: (B, 1, T, T) bool, True = may attend.

        With pos/mask omitted, each row is a single game starting at position 0 (used for generation).
        Returns logits (B, T, vocab_size).
        """
        B, T = idx.shape
        if pos is None:
            pos = torch.arange(T, device=idx.device).expand(B, T)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.ln_f(x))

    def configure_optimizer(self, lr: float, weight_decay: float, betas: tuple[float, float]) -> torch.optim.AdamW:
        # Weight decay on matrices only, not on biases, LayerNorm or embeddings' positional table.
        decay = [p for n, p in self.named_parameters() if p.dim() >= 2 and not n.startswith("pos_emb")]
        no_decay = [p for n, p in self.named_parameters() if p.dim() < 2 or n.startswith("pos_emb")]
        groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=torch.cuda.is_available())


def attention_mask(doc: torch.Tensor) -> torch.Tensor:
    """Block-diagonal causal mask from per-token game ids (B, T): attend only to earlier tokens of the same game.

    Padding has doc id -1, so pad tokens only attend to pad tokens (never an empty row, which would give NaNs).
    """
    T = doc.shape[1]
    causal = torch.ones(T, T, dtype=torch.bool, device=doc.device).tril()
    return ((doc[:, :, None] == doc[:, None, :]) & causal)[:, None]
