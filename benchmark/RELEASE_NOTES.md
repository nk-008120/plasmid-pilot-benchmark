# Benchmark v1: payload retrieval under a holdout

A retrieval benchmark for one question: **does function-based screening need a decomposition step in a
way alignment does not?** Each item hands a method one protein, the payload-bearing ORF as a naive
six-frame caller emitted it, often a fusion, a tagged construct or a two-gene cassette, plus a protein
catalogue, and asks for a ranking of catalogue accessions. The benchmark then removes the payload's
close relatives at six identity thresholds and reads where the best surviving relative lands.

`benchmark/README.md` is the full description. These notes carry only what a reader needs before
downloading anything.

## What is in this release

| | |
|---|---|
| items | **1,102** over four tiers, 173 distinct payloads |
| catalogue | 40,750 proteins, 20,806,731 residues in the 50-5,000 aa window |
| holdout thresholds | 1.00, 0.90, 0.70, 0.50, 0.35, 0.25 |
| payload-fraction strata | `f < 0.50`, `0.50-0.70`, `0.70-0.95`, `f >= 0.95` |
| baselines | exact Smith-Waterman and mean-pooled ESM-2, 576 rows each |

The four tiers are `hand` (20 items, ground truth from the depositor), `corpus` (410, gated alignment,
80% validated), `synth_real` (336, exact boundaries, SwissProt filler) and `synth_random` (336, the same
constructs with random filler).

**`synth_real` and `synth_random` are the same 336 constructs and differ only in their filler residues**:
matched residue for residue on payload fraction. They are two tiers rather than one because the filler
reverses which scorer is hurt: random filler is invisible to a local alignment and is the worst case in
the benchmark for a mean-pooled embedding arm. A result quoted from one of them is not a result about
the other.

## Item ids are namespaced

Because those two tiers share every construct id, **every item is addressed as `tier:construct_id`**,
`synth_real:sc_0001` and `synth_random:sc_0001` are different items. `items.csv`, the `queries.faa`
headers and both baselines use that form, and a submission must return rankings under it.
`MANIFEST.json` records the convention as `construct_ids_namespaced`.

## Two rules that are refusals, not guidance

`pilot/benchmark.py` is the rules as code, and two of them raise rather than returning a number:

- **`pooled_scalar` refuses a scalar pooled across payload-fraction strata.** Pooled `recall@10` at the
  25% holdout is +1.0% while the `f < 0.50` stratum is +40.0, because most items are already saturated
  at depth 10. There is no depth at which this result survives pooling.
- **`mean_rank_phrase` refuses to render an embedding arm's mean rank without its count of
  rank >= 10,000 failures.** Under mean-pooling the rank distribution is bimodal where alignment's is
  not, so the mean stops describing the centre: on the corpus tier at the 90% holdout a `whole` mean
  rank of 1,088 is a median of **1** plus 15 constructs carrying one payload, `P23181`, at about
  29,699 of 40,750. Macro averaging does not rescue it.

Mean rank and MRR are primary, stratified by `f`; `recall@10` is the operational secondary; top-1 is a
column and never a headline. Micro and macro are both always reported, because 410 corpus constructs
carry only 138 distinct payloads and are not independent trials. An item that cannot be scored stays in
the denominator as a miss.

## Which baseline is canonical

**`baseline_alignment.*` and `baseline_embedding.*`**, the unsuffixed pair. They carry
`is_canonical: true` and were measured with the segmenter's holdout mask built **per threshold**.

`baseline_*_prer10.*` is the earlier measurement, kept deliberately. It was made with the mask built
once at the strictest threshold, which inflates the segmented arms' reported *cost* at every looser
holdout; the README's **Erratum, 20 Oct** sets out what it bites and by how much, and the file is that
erratum's evidence. It is three tiers, not four, `synth_random` has no pre-fix measurement, and it
carries `is_canonical: false`. Do not compare against it.

**A note on the baselines' runtime fields.** The alignment baseline's four source runs were timed on two
different machines: hand, corpus and `synth_real` at 14 threads on a 16-logical-processor desktop,
`synth_random` at 3 threads on a laptop. The total is a sum across both and is not a prediction for
either. The ranks are unaffected: a runtime is metadata here, and it is reported with its conditions
for the same reason every other number in this project is.

## The catalogue is attached, not committed

The repository carries `catalogue_accessions.txt` in catalogue order and the checksum below; the FASTA
itself is attached to this release, because it is the one artefact a reader cannot rebuild bit-for-bit
from the repository alone.

```
reference_twilight.fasta
sha256  43b8b3c0b5cb7440ce02387e54bd9b99e3c44f9c5fbb62c94050368da44ca355
        40,750 entries | 40,749 in the 50-5,000 aa window | 20,806,731 residues
```

Verify a copy before scoring anything against it:

```bash
python scripts/52_build_benchmark.py --verify-catalogue <path to fasta>
```

If the residue count differs, the significance floor differs, which makes it a **different benchmark**
rather than a slightly different one. The script checks all four quantities and says which failed.

The catalogue is derived from UniProt Swiss-Prot, distributed under CC BY 4.0; attribution travels with
the FASTA. This repository's own contents are MIT. The Swiss-Prot release label was not recorded at
download time: the sha256 of the source archive, in `MANIFEST.json`, is what pins it.

## The two scoring routes are checked, not assumed

`55_crosscheck_benchmark.py` scores the published queries for the hand tier and compares **every cell**
against the measurement script's own ranks: **120 of 120 identical** on the build published here. It
exits non-zero on a single mismatch, and, since 22 Oct, also when it compares nothing, after it was
caught reporting agreement on a run in which no cell had matched at all.

## What this does not establish

A reviewer's score and this project's should differ by method and by nothing else; that is the point of
shipping the rules as code rather than a leaderboard. But `synth_real` and `synth_random` are
constructed conditions and are bounds on the real case, not measurements of it; the `hand` and `corpus`
tiers are the measured ends. The `oracle` arm uses the label and is a ceiling, never a method. And every
number here is one catalogue, one holdout construction and two scorers. The working repository
keeps the register of what each of those has changed and the errata that found it; it is not
part of this release, and nothing in this benchmark depends on reading it.
