# Encrypted peptides: antimicrobial design from proteomes that were never antimicrobial

**AMP Challenge 2027 — method abstract**

## Method

We designed the library from **encrypted peptides**: contiguous windows of ordinary
proteomes that carry no antimicrobial annotation. The corpus holds 100,348 such
peptides, mined with parent-protein identity and residue coordinates retained for 97%
of them, drawn from human, archaeal, venom, prion and extinct-organism proteomes.

The premise is that antimicrobial sequence space is substantially larger than the
databases that catalogue it. The competition's own constraint makes that measurable:
**91.7% of GRAMPA — the standard public AMP training corpus — exact-matches the
39,448-sequence reference set, against 0.435% of our encrypted peptides.** Teams
training on curated AMP databases are largely training on sequences they are not
permitted to submit.

**Generator.** A 4.76M-parameter decoder-only transformer (6 layers, 256 d, 8 heads),
trained from scratch on the union of 99,911 encrypted peptides and the 39,448 reference
antibacterials. Every sequence is prefixed with three conditioning tokens — source,
length bin, charge bin. The encrypted half supplies novelty; the reference half supplies
the physicochemical grammar of a working antimicrobial. Sampling draws each conditioning
cell from a blend of the reference histogram and a flat one, across a temperature ladder;
the flattening is what buys mode coverage. Validity, uniqueness and exact-match exclusion
are enforced at sample time, before a sequence can enter the library.

**Top-100.** Not model output. Drawn from encrypted peptides carrying real published MIC
measurements, ranked on **measured breadth** — the fraction of panel strains reaching
MIC ≤ 16 µM — because Phase 2 scores the mean per-peptide Success Rate over 25 randomly
drawn candidates. Candidates must have been tested on at least 5 panel strains and must
clear the similarity constraint, which we enforce at 0.78 rather than the rule's 0.8 so
the submission is not sitting exactly on a threshold in a live reference file.

Because each category score is a *mean* over the drawn peptides, each is linear in the
selection, and the mean of the four panel-scored categories is maximised exactly by
taking the top 100 on `(1.25·Gram+ + 1.75·Gram− + 1.00·MDR) / 4` — Broad-Spectrum being
identically `0.25·Gram+ + 0.75·Gram−` at 5 and 15 of 20 strains. Those coefficients are
arithmetic from the panel composition; there is nothing tuned in the objective. A
hemolysis screen enters as a weak veto and a tiebreak, at the strength justified below.

## Results

Library, scored with the organizers' `seqme` metric suite against real AMPs and against
character-permuted AMPs as a floor. **All rows are at n = 3,000**, the sample size the
reference rows are computed at — FKEA counts effective modes and therefore grows with set
size, so rows measured at different n are not comparable:

| | Diversity ↑ | FKEA ↑ | FBD(UniProt) ↓ | FBD(AMPs) ↓ | Novelty ↑ | AuthPct ↑ | Conformity ↑ |
|---|---|---|---|---|---|---|---|
| AMPs (real) | 0.854 | 759 | 2.33 | 0.041 | 0.775 | 0.588 | 0.496 |
| AMPs (permuted) | 0.857 | 610 | 2.79 | 0.727 | 0.999 | 0.801 | 0.509 |
| uniform draw from our corpus | 0.804 | 428 | 3.13 | 1.919 | 1.000 | 0.884 | 0.329 |
| **submitted library** | **0.846** | **947** | **2.645** | **0.235** | **1.000** | **0.857** | **0.470** |

This row is measured on `generate/library.fasta` as produced by `uv run generate` in this
repository — the file the organizers will score. We flag that as a methodological point:
our research pipeline builds a second library from a pre-generated pool that shares only
1,052 of 50,000 sequences with the shipped one, and its metrics differ materially
(FKEA 908, FBD(AMPs) 0.202). Set-level metrics must be measured on the artifact that is
actually submitted.

A uniform draw from the raw corpus scores *below character-permuted AMPs* on every
realism metric — a result we report because it is the reason the generator exists.
Conditional generation raises FKEA from 428 to 947, well above real AMPs at the same
sample size, and FBD(AMPs) from 1.919 to 0.235 — an order of magnitude, and 3.1× better
than permuted AMPs — while Novelty stays at 1.000. The cost is ConformityScore, 0.470
against 0.496 for real AMPs: we sit just inside the reference distribution rather than above it.
We report the whole row rather than a scalar because the components genuinely conflict
and any single number would hide which side of that trade was paid.

Submitted top-100, measured (not predicted) over the 11 panel strains we hold data for:
**Broad-Spectrum 70.5%, Gram-negative 75.6%, MDR 71.0%, Gram-positive 55.2%**; median
MIC50 8.0 µM; mean 10.0 of 11 strains tested per peptide. The oracle ceiling on the same
measured population is 77.0% Broad-Spectrum, so the list sits at roughly 92% of what
perfect foresight could select. All 100 carry real measurements; none is a prediction.

## Three findings we would flag to the organizers

**Activity predictors do not transfer to de novo peptides — including ours, and
including the state of the art.** The 506 wet-lab MICs on 46 diffusion-generated peptides
in `organizer_mic.csv` have zero overlap with our training data and are, as far as we can
establish, the only published held-out test of this kind. Scores are converted to
within-strain percentile ranks before pooling; intervals bootstrap over peptides.

| model | pooled AUC | 95% CI |
|---|---|---|
| **APEX** — 40-model ensemble, from the laboratory that runs this panel | **0.585** | [0.503, 0.663] |
| net charge — **no training** | 0.502 | [0.398, 0.603] |
| hydrophobic fraction — **no training** | 0.487 | [0.376, 0.607] |
| ours — XGBoost + ESM-2 on 59k MIC rows, ~0.90 under cluster-grouped CV | 0.485 | [0.405, 0.569] |
| sequence length — control | 0.464 | [0.357, 0.568] |

APEX is the only method above chance. **Our supervised model does not beat net charge and
is statistically indistinguishable from ranking by length.** We report the untrained
baselines deliberately: against chance alone a 0.485 reads as bad luck, but against net
charge it reads as nothing transferable having been learned.

We attribute this to the training distribution rather than to any single model. Everyone
in this field trains on natural and curated AMPs and deploys on generated sequences.
Cluster-grouped cross-validation, the standard rigour bar, controls for near-duplicate
leakage but not for *provenance*: a model can reach 0.9 by learning which laboratory a
peptide came from. The same signature appears in our hemolysis screen — 0.910 grouped by
sequence cluster, **0.686** grouped by citing publication. This is why our top-100 is
measurements rather than predictions, and we would encourage the organizers to treat a
de novo held-out set as a standard reporting requirement.

**The novelty rule disqualifies most of the natural AMP density.** Sampling 1,500
reference AMPs against all 39,448, the median nearest-neighbour Levenshtein ratio *within
the reference set* is 0.905, and **64.1% of real, wet-lab-validated AMPs would themselves
fail the top-100 similarity rule against another reference AMP.** Any design strategy
built on editing known AMPs is searching a space where most of the natural density is
already excluded.

**Phase 2 can resolve differences above ~3.2%.** Simulating the 25-of-100 uniform draw
puts the standard error of a realised team score at about 1.1%, so two teams whose true
mean Success Rates differ by less than roughly 3.2% will be ordered largely by chance.
We report this because it bounds how much of the final ranking is measurement.

## Limitations

- We hold measurements for **11 of the 20 panel strains** (Gram-negative 7/15, MDR 4/8).
  Every Success Rate above is over that covered subset and should be read as such.
- **Gram-positive activity remains our weakest axis at 55.2%**, against an 87.8% oracle
  ceiling. It does not move under any reweighting of the selection objective —
  *E. faecalis* ATCC 700802 and *S. aureus* BAA-1556 are thin across the entire candidate
  pool. It is a gap in the data, not in the ranking rule, and the only thing that has
  moved it was adding data: ingesting 880 archaeasin measurements took it from 51.2% to
  55.2%.
- A species-level label expansion (30,597 GRAMPA rows, taking nominal strain coverage
  from 11/20 to 20/20) **did not work**: built without leakage, it degraded top-100
  selection from 55.8% to 51.7%. We report it as a negative result. Cross-strain breadth
  transfer within our own measured peptides is strong (leave-one-strain-out mean AUC
  0.914); transfer from GRAMPA's peptide population to ours is not.
- Measurement noise floor in the training data is 0.58 log2; 31% of repeat MICs disagree
  by more than 2×. No model here is tuned past that.

## Reproducibility

MIT licensed. `uv`-managed with a committed lockfile and pinned Python. `uv run generate`
is CPU-only with a fixed seed and produces byte-identical output across runs, verified.
All training data is public; the encrypted peptide corpus is released alongside this
submission.
