"""AMP Challenge 2027 entry point: `uv run generate`.

Writes generate/library.fasta (50,000) and generate/top.fasta (100).

Determinism
-----------
The organizers run this twice and compare bytes, so every source of ordering
is pinned: torch runs on CPU with a seeded generator, all set operations are
funnelled through sorted(), and nothing that reaches a file depends on dict or
set iteration order. Sampling is the only stochastic step and it draws from one
explicitly seeded torch.Generator.

Method in brief
---------------
A 4.8M-parameter conditional transformer LM, trained from scratch on 99,911
encrypted peptides (contiguous fragments of non-antimicrobial proteomes, which
overlap the competition reference set at 0.435%) plus 39,448 reference
antibacterials. Conditioning is on (length bin, charge bin, source), which lets
the sampler match the reference physchem histogram while spreading across the
conditioning grid for mode coverage.

The top-100 is not model output. It is drawn from peptides carrying real,
published MIC measurements against the panel, ranked by measured breadth --
the number of strains at MIC <= 16 uM -- because the competition scores the
mean per-peptide Success Rate over 25 randomly drawn candidates. The ranking
table is shipped in data/top_candidates.csv with its provenance.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

from encrypted_amp.model import (
    PeptideLM,
    Config,
    CANON,
    build_plan,
    chg_bin,
    len_bin,
    net_charge,
)

HERE = Path(__file__).resolve().parent
CKPT = HERE / "checkpoint" / "generator.pt"
REFERENCE = HERE / "data" / "antibacterial.fasta"
CANDIDATES = HERE / "data" / "top_candidates.csv"

MIN_LEN, MAX_LEN = 8, 50

# The competition rule: a top-100 sequence fails if Levenshtein.ratio against
# any reference sequence is > 0.8.
SIM_THRESHOLD = 0.8
# What selection actually enforces. Held one notch stricter so the submission
# is not sitting exactly on the rule: the reference FASTA is a file in a live
# repository, and a sequence at a ratio of precisely 0.8000 passes only because
# the validator's comparison is strictly `>`.
SIM_MARGIN = 0.78

# Sampling configuration, fixed by Phase 1 scoring rather than by argument.
# See the repository README for the metric table these settings were chosen on.
MIX_UNIFORM = 0.25
TEMPS = (0.85, 1.0, 1.15)
AMP_FRACTION = 0.8
TOP_P = 0.95
# Draw a surplus, since duplicates and reference-set hits are dropped. Measured
# keep rate on a 75k draw was 93.3% (8 invalid, 1,305 duplicate, 3,733 already
# in the reference set), so 1.3 clears 50,000 with ~21% headroom while cutting
# roughly a fifth off a run that the validator executes twice. Falling short
# raises rather than silently truncating, so the margin fails loudly.
OVERSAMPLE = 1.3


def read_fasta(path: Path) -> list[str]:
    seqs: list[str] = []
    cur: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
                cur = []
        else:
            cur.append(line.upper())
    if cur:
        seqs.append("".join(cur))
    return seqs


def write_fasta(seqs: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for i, s in enumerate(seqs, start=1):
            fh.write(f">seq{i}\n{s}\n")


def valid(seq: str) -> bool:
    return MIN_LEN <= len(seq) <= MAX_LEN and not set(seq) - CANON


def load_model() -> PeptideLM:
    blob = torch.load(CKPT, map_location="cpu", weights_only=True)
    model = PeptideLM(Config(**blob["cfg"]))
    model.load_state_dict(blob["model"])
    model.eval()
    return model


def generate(n_sequences: int, *, seed: int = 42, length: int = MAX_LEN) -> list[str]:
    """Sample, filter, and return exactly n_sequences unique valid peptides."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    model = load_model()
    gen = torch.Generator().manual_seed(seed)
    reference = set(read_fasta(REFERENCE))

    plan = build_plan(int(n_sequences * OVERSAMPLE), MIX_UNIFORM)
    seen: set[str] = set()
    kept: list[str] = []

    for (lb, cb) in sorted(plan):
        want = plan[(lb, cb)]
        per_temp = max(1, want // len(TEMPS))
        for ti, temp in enumerate(TEMPS):
            source = 1 if ((lb * 7 + cb * 3 + ti) % 10) < int(AMP_FRACTION * 10) else 0
            remaining = per_temp
            while remaining > 0:
                k = min(512, remaining)
                for s in model.sample(
                    k, source, lb, cb, temperature=temp, top_p=TOP_P,
                    max_len=min(length, MAX_LEN), generator=gen,
                ):
                    if valid(s) and s not in reference and s not in seen:
                        seen.add(s)
                        kept.append(s)
                remaining -= k

    # Guaranteed slots for peptides with real measurements: they are the only
    # library members eligible for the top-100, so they must be present before
    # the library is truncated to size.
    anchors = [row["sequence"] for row in _read_candidates()]
    anchors = [s for s in anchors if valid(s) and s not in reference]

    # `kept` is already in sampling order (grid cell, then temperature), which
    # correlates with length and charge, so truncating it directly would bias
    # the library toward whichever cells happen to come first. A fixed-seed
    # permutation over a sorted list is unbiased and still byte-reproducible.
    kept_sorted = sorted(kept)
    perm = _rng_permutation(len(kept_sorted), seed)
    library = list(dict.fromkeys(anchors + [kept_sorted[i] for i in perm]))
    if len(library) < n_sequences:
        raise RuntimeError(
            f"only {len(library):,} unique sequences after filtering; "
            f"raise OVERSAMPLE (currently {OVERSAMPLE})"
        )
    return library[:n_sequences]


def _rng_permutation(n: int, seed: int) -> list[int]:
    """Deterministic permutation without adding a numpy RNG dependency here."""
    import random

    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return idx


def _read_candidates() -> list[dict]:
    with CANDIDATES.open() as fh:
        return list(csv.DictReader(fh))


def score(sequences: list[str]) -> list[float]:
    """Rank score, higher is better. Measured breadth where known, else 0."""
    table = {row["sequence"]: float(row["rank_score"]) for row in _read_candidates()}
    return [table.get(s, 0.0) for s in sequences]


def select_top(library: list[str], top_k: int) -> list[str]:
    """Top-k by measured breadth, enforcing the Levenshtein constraint."""
    import Levenshtein

    reference = read_fasta(REFERENCE)
    in_lib = set(library)
    ranked = [r for r in _read_candidates() if r["sequence"] in in_lib]
    ranked.sort(key=lambda r: (-float(r["rank_score"]), r["sequence"]))

    chosen: list[str] = []
    for row in ranked:
        if len(chosen) >= top_k:
            break
        seq = row["sequence"]
        # score_cutoff makes rapidfuzz bail early on hopeless pairs. Semantics
        # are unchanged: a true ratio above the threshold is still returned
        # exactly, and anything below it returns 0.0, which also fails the test.
        if all(
            Levenshtein.ratio(seq, ref, score_cutoff=SIM_MARGIN) <= SIM_MARGIN
            for ref in reference
        ):
            chosen.append(seq)

    if len(chosen) < top_k:
        raise RuntimeError(f"only {len(chosen)} candidates passed the similarity filter")
    return chosen


def main() -> None:
    entry_point = Path(sys.argv[0]).stem

    ap = argparse.ArgumentParser(description="Generate the AMP Challenge 2027 submission.")
    ap.add_argument("--n-sequences", type=int, default=50_000)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--length", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(entry_point)

    sequences = generate(args.n_sequences, seed=args.seed, length=args.length)
    library_path = out_dir / "library.fasta"
    write_fasta(sequences, library_path)
    print(f"Generated {len(sequences)} sequences → {library_path}")

    top = select_top(sequences, args.top_k)
    top_path = out_dir / "top.fasta"
    write_fasta(top, top_path)
    print(f"Top {len(top)} sequences → {top_path}")


if __name__ == "__main__":
    main()
