"""Conditional peptide language model — architecture and sampling.

Self-contained by design: the submission depends only on torch, numpy and
Levenshtein, so nothing here imports pandas, seqme, or the research tree. The
architecture must match the training definition exactly or the checkpoint will
not load.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
GRID = HERE / "data" / "reference_grid.json"

AA = "ACDEFGHIKLMNPQRSTVWY"
CANON = set(AA)
MAX_LEN = 50

LEN_BINS = [8, 12, 16, 20, 25, 30, 40, 51]
CHG_BINS = [-math.inf, 0, 2, 4, 6, 8, 10, math.inf]
N_LEN_BINS = len(LEN_BINS) - 1
N_CHG_BINS = len(CHG_BINS) - 1
N_SOURCES = 2

PAD, BOS, EOS = 0, 1, 2
AA_OFFSET = 3
SRC_OFFSET = AA_OFFSET + len(AA)
LEN_OFFSET = SRC_OFFSET + N_SOURCES
CHG_OFFSET = LEN_OFFSET + N_LEN_BINS
VOCAB = CHG_OFFSET + N_CHG_BINS

AA_TO_ID = {a: AA_OFFSET + i for i, a in enumerate(AA)}
ID_TO_AA = {v: k for k, v in AA_TO_ID.items()}

CTX = 3 + 1 + MAX_LEN + 1


def net_charge(seq: str) -> float:
    return (seq.count("K") + seq.count("R") + 0.1 * seq.count("H")
            - seq.count("D") - seq.count("E"))


def len_bin(n: int) -> int:
    for i in range(N_LEN_BINS):
        if LEN_BINS[i] <= n < LEN_BINS[i + 1]:
            return i
    return N_LEN_BINS - 1 if n >= LEN_BINS[-1] else 0


def chg_bin(c: float) -> int:
    """Must reproduce np.digitize(c, CHG_BINS[1:-1]) exactly.

    Training encoded the conditioning tokens with numpy's left-closed
    convention, so a half-open interval here would shift every charge token by
    one bin at generation time and the conditioning would silently mean
    something the model was never trained on.
    """
    n = sum(1 for edge in CHG_BINS[1:-1] if edge <= c)
    return min(max(n, 0), N_CHG_BINS - 1)


def build_plan(n: int, mix_uniform: float, min_cell: int = 1) -> dict[tuple[int, int], int]:
    """Sequences to draw per (length bin, charge bin) conditioning cell.

    Blends the reference AMP histogram — which ConformityScore is computed
    against — with a flat histogram, which spreads the pool across more of the
    conditioning grid and lifts the mode-count metrics.
    """
    with GRID.open() as fh:
        raw = json.load(fh)
    cells = [(int(k.split(",")[0]), int(k.split(",")[1])) for k in raw]
    probs = [float(v) for v in raw.values()]

    flat = 1.0 / len(cells)
    weights = [(1 - mix_uniform) * p + mix_uniform * flat for p in probs]
    total = sum(weights)
    return {c: max(min_cell, int(round(w / total * n))) for c, w in zip(cells, weights)}


@dataclass
class Config:
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.1


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = nn.MultiheadAttention(
            cfg.n_embd, cfg.n_head, dropout=cfg.dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class PeptideLM(nn.Module):
    def __init__(self, cfg: Config = Config()):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(VOCAB, cfg.n_embd)
        self.pos = nn.Embedding(CTX, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, VOCAB, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, idx):
        t = idx.shape[1]
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos))
        mask = torch.triu(torch.ones(t, t, device=idx.device, dtype=torch.bool), diagonal=1)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def sample(
        self,
        n: int,
        source: int,
        lb: int,
        cb: int,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_len: int = MAX_LEN,
        generator: torch.Generator | None = None,
    ) -> list[str]:
        self.eval()
        device = next(self.parameters()).device
        prefix = torch.tensor(
            [[SRC_OFFSET + source, LEN_OFFSET + lb, CHG_OFFSET + cb, BOS]], device=device
        ).repeat(n, 1)
        done = torch.zeros(n, dtype=torch.bool, device=device)
        steps: list = []

        banned = torch.zeros(VOCAB, dtype=torch.bool, device=device)
        banned[PAD] = banned[BOS] = True
        banned[SRC_OFFSET:] = True

        for _ in range(max_len + 1):
            logits = self(prefix)[:, -1, :] / max(temperature, 1e-6)
            # A finished row keeps getting extended below, so its prefix leaves
            # the training distribution and can go non-finite; a NaN row makes
            # multinomial return an arbitrary index, which then decodes as a
            # conditioning token rather than a residue.
            logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
            logits = logits.masked_fill(banned, float("-inf"))
            probs = F.softmax(logits, dim=-1)

            if top_p < 1.0:
                srt, idx = torch.sort(probs, descending=True, dim=-1)
                cum = srt.cumsum(dim=-1)
                keep = cum - srt <= top_p
                keep[:, 0] = True
                srt = srt * keep
                srt = srt / srt.sum(dim=-1, keepdim=True)
                nxt = idx.gather(-1, torch.multinomial(srt, 1, generator=generator))
            else:
                nxt = torch.multinomial(probs, 1, generator=generator)

            # Bookkeeping is vectorised: a per-element .item() would force one
            # device sync per sequence per token (3M syncs for a 60k draw). The
            # token fed back into the prefix is unchanged, so the RNG stream and
            # therefore the output are bit-identical to the scalar version.
            nxt = nxt.squeeze(-1)
            # Freeze finished rows on EOS so their prefixes stay in-distribution.
            # Rows are independent under causal attention, so this cannot change
            # any sequence still being generated.
            nxt = torch.where(done, torch.full_like(nxt, EOS), nxt)
            steps.append(nxt)
            done = done | (nxt == EOS)
            if bool(done.all()):
                break
            prefix = torch.cat([prefix, nxt.unsqueeze(-1)], dim=1)

        drawn = torch.stack(steps, dim=1).cpu().tolist()
        out = []
        for row in drawn:
            chars = []
            for tok in row:
                if tok == EOS or tok not in ID_TO_AA:
                    break   # defence in depth: never decode a non-residue token
                chars.append(ID_TO_AA[tok])
            out.append("".join(chars))
        return out
