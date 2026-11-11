import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from photonic.calibration import CalibrationCollector


@dataclass
class ModelSpec:
    name: str
    d_model: int
    n_layer: int
    n_head: int
    # None preserves the original architecture: d_ff = 4 * d_model
    d_ff: Optional[int] = None

    @property
    def effective_d_ff(self):
        return 4 * self.d_model if self.d_ff is None else int(self.d_ff)


def linear_dispatch(
    x,
    layer,
    name,
    engine=None,
    calibration=None,
    collector=None,
):
    if collector is not None:
        collector.add(name, x)
    if engine is None:
        return layer(x)
    if calibration is None or name not in calibration:
        raise KeyError(f'Missing calibration for {name}')
    return engine.forward(
        x,
        layer.weight,
        layer.bias,
        name,
        calibration[name],
    )


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, idx):
        super().__init__()
        assert d_model % n_head == 0, (
            f'd_model={d_model} must be divisible by n_head={n_head}'
        )
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.idx = idx
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x,
        engine=None,
        calibration=None,
        collector=None,
    ):
        B, T, C = x.shape
        qkv = linear_dispatch(
            x,
            self.qkv,
            f'block{self.idx}.attn.qkv',
            engine,
            calibration,
            collector,
        )
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(
            B, T, self.n_head, self.head_dim
        ).transpose(1, 2)
        k = k.view(
            B, T, self.n_head, self.head_dim
        ).transpose(1, 2)
        v = v.view(
            B, T, self.n_head, self.head_dim
        ).transpose(1, 2)

        s = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        mask = torch.triu(
            torch.ones(
                T,
                T,
                dtype=torch.bool,
                device=x.device,
            ),
            diagonal=1,
        )
        s = s.masked_fill(mask, float('-inf'))
        a = F.softmax(s, dim=-1)
        y = a @ v

        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        return linear_dispatch(
            y,
            self.out_proj,
            f'block{self.idx}.attn.out',
            engine,
            calibration,
            collector,
        )


class MLP(nn.Module):
    def __init__(self, d_model, d_ff, idx):
        super().__init__()
        self.idx = idx
        self.d_ff = int(d_ff)
        self.fc1 = nn.Linear(d_model, self.d_ff)
        self.fc2 = nn.Linear(self.d_ff, d_model)

    def forward(
        self,
        x,
        engine=None,
        calibration=None,
        collector=None,
    ):
        x = linear_dispatch(
            x,
            self.fc1,
            f'block{self.idx}.ffn.fc1',
            engine,
            calibration,
            collector,
        )
        x = F.gelu(x)
        return linear_dispatch(
            x,
            self.fc2,
            f'block{self.idx}.ffn.fc2',
            engine,
            calibration,
            collector,
        )


class Block(nn.Module):
    def __init__(self, d_model, n_head, d_ff, idx):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model,
            n_head,
            idx,
        )
        self.mlp = MLP(
            d_model,
            d_ff,
            idx,
        )

    def forward(
        self,
        x,
        engine=None,
        calibration=None,
        collector=None,
    ):
        x = x + self.attn(
            self.ln1(x),
            engine,
            calibration,
            collector,
        )
        x = x + self.mlp(
            self.ln2(x),
            engine,
            calibration,
            collector,
        )
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        spec,
    ):
        super().__init__()
        self.block_size = block_size
        self.spec = spec
        d_ff = spec.effective_d_ff

        self.token_embedding = nn.Embedding(
            vocab_size,
            spec.d_model,
        )
        self.position_embedding = nn.Embedding(
            block_size,
            spec.d_model,
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    spec.d_model,
                    spec.n_head,
                    d_ff,
                    i,
                )
                for i in range(spec.n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(spec.d_model)
        self.lm_head = nn.Linear(
            spec.d_model,
            vocab_size,
            bias=False,
        )

        # Weight tying, unchanged from the original model.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        idx,
        targets=None,
        engine=None,
        calibration=None,
        collector=None,
        photonic_lm_head=True,
    ):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(
                'Sequence exceeds block_size'
            )

        pos = torch.arange(
            T,
            device=idx.device,
        )

        x = (
            self.token_embedding(idx)
            + self.position_embedding(pos)[None, :, :]
        )

        for block in self.blocks:
            x = block(
                x,
                engine,
                calibration,
                collector,
            )

        x = self.ln_f(x)

        if engine is not None and photonic_lm_head:
            logits = linear_dispatch(
                x,
                self.lm_head,
                'lm_head',
                engine,
                calibration,
                collector,
            )
        else:
            if collector is not None:
                collector.add('lm_head', x)
            logits = self.lm_head(x)

        loss = (
            None
            if targets is None
            else F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )
        )

        return logits, loss


def total_parameter_count(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )


def non_embedding_parameter_count(model):
    excluded = {
        id(model.token_embedding.weight),
        id(model.position_embedding.weight),
    }
    seen = set()
    total = 0

    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        if id(p) not in excluded:
            total += p.numel()

    return total
