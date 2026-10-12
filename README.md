# Encrypted Peptipedia — AMP Challenge 2027 submission

A conditional peptide language model trained on **encrypted peptides**: contiguous
fragments of proteomes that were never annotated as antimicrobial. The premise is that
antimicrobial sequence space is much larger than the AMP databases that catalogue it,
and that the unexplored part is reachable by mining ordinary proteins.

```bash
uv sync
uv run generate
```

Writes `generate/library.fasta` (50,000) and `generate/top.fasta` (100). Fixed default
seed; two runs are byte-identical.

| | |
|---|---|
| Corpus collision with the competition reference set | **0.435%**, against **91.7%** for GRAMPA |
| Top-100 carrying real published MIC measurements | **100 / 100** |
| Selection rule, held-out cluster-grouped folds | **11.4% → 55.4%** mean Success Rate |
| Our own ranker on organizer-measured de novo peptides | **0.485 AUC — no measurable skill** |

That last row is not a typo. See [A model result we would flag](#a-model-result-we-would-flag).

## Why this corpus

The competition reference set (`data/antibacterial.fasta`, 39,448 sequences) is the
union of the standard AMP databases. Anything a model trained on those databases
produces is likely to collide with it — and the rules forbid exact matches in the
library and cap top-100 similarity at 0.8 Levenshtein ratio.

| corpus | exact-matches the reference set |
|---|---|
| GRAMPA (DBAASP + DRAMP + APD + YADAMP + DADP) | **91.7%** |
| our encrypted peptides | **0.435%** |

Teams training on standard AMP databases are largely training on sequences they are not
permitted to submit. The encrypted corpus does not have that problem: 99,911 of 100,348
sequences survive the exact-match filter, and all 100,348 are natively 8–50 residues and
fully canonical.

## Method

**Generator.** A 4.8M-parameter decoder-only transformer (6 layers, 256 d, 8 heads),
trained from scratch on 99,911 encrypted peptides plus 39,448 reference antibacterials.
Each sequence is prefixed with three conditioning tokens — source, length bin, charge bin
— so sampling can steer both the physicochemical profile and how much of the corpus it
draws from.

Sampling draws each conditioning cell in proportion to a blend of the reference AMP
histogram and a flat histogram, across a temperature ladder. The blend is the explicit
trade in this submission: the reference histogram is what `ConformityScore` is computed
against, while flattening and the temperature ladder widen the mode coverage that
`FKEA` and `Diversity` reward. Every sample is filtered for validity, uniqueness, and
exact matches against the reference set before it can enter the library.

**Top-100.** Not model output. Drawn from encrypted peptides that carry real, published
MIC measurements against the competition panel, ranked by **measured breadth** — the
number of strains at MIC ≤ 16 µM.

That choice follows directly from the scoring rule. Success Rate is a per-peptide
quantity: the percentage of panel strains where *that* peptide clears 16 µM, averaged
over the 25 peptides drawn at random from the list. Ranking by "is it active anywhere"
optimises a different quantity and fills the list with peptides active on one strain. On
held-out cluster-grouped folds the difference is not marginal:

| selection rule | mean Success Rate (covered panel) |
|---|---|
| random ordering | 11.4% |
| best single strain | 45.3% |
| **mean across the panel** | **55.4%** |
| oracle (perfect foresight) | 77.0% |

Those are *in-distribution* numbers and we do not rely on them: the submitted top-100 is
100/100 peptides carrying real measurements, and the model score enters only as a fourth
tiebreak. See "A model result we would flag" below for why that matters.

**Ranking objective.** Four of the five categories are scored from this panel and each
team score is a *mean* over the 25 drawn peptides, so each is linear in the selection and
the mean of the four is maximised exactly by taking the top 100 on a per-peptide score.
Because Broad-Spectrum is identically `0.25·Gram+ + 0.75·Gram−` (5 and 15 of the 20
strains), that score is `(1.25·Gram+ + 1.75·Gram− + 1.00·MDR) / 4`. These coefficients
are arithmetic from the panel composition, not tuned hyperparameters.

**Hemolysis screen.** Candidates are additionally filtered on predicted red-blood-cell
safety, since the same 25 peptides are scored for Optimal Selectivity (HC50 / MIC50)
whether or not a team models it. Labels are 476 sequences typed out of DRAMP free text,
keeping HC50/LC50/EC50 midpoints separate from the stricter MHC family rather than
converting between them.

We report two numbers for it, because only one of them is trustworthy. Grouping
cross-validation folds on sequence cluster gives OOF AUC **0.910**. Grouping them on the
*citing publication* instead — DRAMP records a PubMed id for 782 of 820 hemolysis
records, across 195 papers — gives **0.686 [0.630, 0.741]**. Cluster grouping stops a
model recalling a near-duplicate; it does nothing about a model recalling which
laboratory a peptide came from. The screen is therefore applied as a weak veto
(P(safe) ≥ 0.25) plus a tiebreak, not as a hard filter: at the stricter threshold it was
removing measured potency on the strength of a 0.686 prediction.

## Honest limits

- We hold measurements for **11 of the 20** panel strains, and 4 of the 8 MDR isolates.
  Every Success Rate quoted above is over the covered subset and should be read as such.
  The uncovered strains rest on resistance transfer (Spearman 0.842 susceptible →
  colistin-resistant *E. coli*, 0.742 → MRSA), which is an assumption, not a measurement.
- The *E. faecalis* ATCC 700802 head is weak (AUC 0.559) — only 1.8% of its rows are
  positive. Top-100 selection uses measured breadth rather than that head's predictions,
  which sidesteps but does not solve the gap.
- Measurement noise floor is 0.58 log2; 31% of repeat MIC measurements in the training
  data disagree by more than 2×. No model here is tuned past that.
- The hemolysis screen transfers from reference AMPs onto encrypted peptides: only 22 of
  its labelled sequences are from our corpus. It is a screen, not a measurement.

## A model result we would flag

`data/07_bench/organizer_mic.csv` holds 506 wet-lab MICs on 46 diffusion-generated
peptides, measured outside our pipeline, with zero overlap against our training tables.
That is the population this competition actually tests. Our per-strain ranker, which
scores ~0.90 AUC on cluster-grouped cross-validation, scores **0.485 [0.405, 0.575]**
there — no measurable skill. Scored on the identical peptides, **APEX** — the 40-model
ensemble from the laboratory that runs this panel — reaches **0.585 [0.503, 0.663]**,
better than ours (paired difference +0.099, P = 97.6%) but only marginally above chance
itself.

We read this as a field-wide generalisation gap rather than one broken model: sequence →
activity predictors trained on curated natural AMPs retain little ranking skill on de
novo peptides. It is also why the top-100 here is built from measurements rather than
predictions.

## Training data

| source | what | licence |
|---|---|---|
| Peptipedia v2 + UniProt/NCBI proteomes | 100,348 encrypted peptides | CC-BY / public |
| GRAMPA | 44,042 MIC records, 6,160 sequences | public |
| de la Fuente lab publications | 15,449 MIC records incl. confirmed-inactive negatives | published supplements |
| Torres et al., *Nature Microbiology* 2025 (archaeasins) | 880 MIC records, 80 peptides, full 11-strain panel | published supplement |
| DRAMP | 820 hemolysis records | public |
| `data/antibacterial.fasta` | competition reference set | supplied by organizers |

Every MIC ingested is checked to lie on the 2-fold broth-microdilution grid before it is
accepted. Several supplements in this literature ship APEX *predictions* in exactly the
shape of measurements — one 4,618-row table on the full panel is entirely model output,
with 4,617 distinct values — and the grid check is what separates them. `>64 µM` is
carried as censored, never as the number 64.

No proprietary or non-public data is used. Full provenance, parsers, and QC gates are in
the research repository accompanying this submission.

## Related

- **[AMP structure atlas](https://github.com/aarnavbhat23/amp-top100-atlas)** —
  [live explorer](https://aarnavbhat23.github.io/amp-top100-atlas/) for Boltz-2.1
  predicted peptide structures, with residue confidence and PAE matrices.
- **[Wet-lab false-positive audit](https://github.com/aarnavbhat23/spear-wetlab-false-positive-audit)** —
  the adversarial protocol used to check whether assay rows in this literature
  carry sequence-specific evidence.

## Licence

MIT. See `LICENSE`.
