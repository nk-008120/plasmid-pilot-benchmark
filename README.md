# Payload retrieval under a holdout - A benchmark

This directory is the part of `plasmid-pilot` another person can run. It publishes the items, the answer
key, and **the metric rules as code** (`src/pilot/benchmark.py`), so a result measured here and a result
measured by this project differ by method and by nothing else.

Why that matters more than usual: every number in this repository depends on which catalogue it was
measured against, which holdout was applied, and which metric was read, and this project has got each of
those three wrong at least once, and the errata are kept in the working repository rather than
here. A benchmark that shipped only a score
would be inviting the same mistakes again, so this one ships the conditions.

```
benchmark/v1/
  items.csv                  1,102 items: id, tier, payload accession, ORF length, oracle length,
                             f, stratum, and where each one came from (see Provenance)
  queries.faa                one protein per item: what a method is handed
  homologs.csv               the answer key: 51,093 rows over 173 payloads, with identity
  catalogue_accessions.txt   40,750 accessions, in catalogue order (the column order for a matrix)
  MANIFEST.json              every rule, count, threshold and checksum this build used
  baseline_alignment.json    this project's own arms under these rules, and the denominator note
  baseline_alignment.csv     the same 576 rows, flat
  baseline_*_prer10.json     the pre-fix measurement, kept as the erratum's evidence
  crosscheck_hand_whole.json 120 of 120 cells identical to scripts/10_twilight.py's own ranks

benchmark/
  presentation.pptx          80 slides on why the task has this shape (see The presentation)
```

## The task

For each item you are handed **one protein**, the payload-bearing ORF as a naive six-frame caller emitted
it, which is often a fusion, a tagged construct or a two-gene cassette, and the catalogue. Return a
ranking of catalogue accessions.

The benchmark then removes the payload's close relatives at each of six identity thresholds and asks where
the best **surviving** relative landed. At a 0.25 holdout everything above 25% identity to the payload is
gone, so success means retrieving a genuine but remote homolog. That is the twilight zone, and it is the
regime the whole project is about.

Nothing in the task requires knowing what a plasmid is.

## What is deliberately not in `items.csv`

No archetype, no segment boundaries, no payload span, no tag positions, nothing a decomposition step is
supposed to discover. `queries.faa` holds the whole ORF, not the payload.

`f` **is** published, and it needs a word of explanation because it is derived from the label. `f =
oracle_span_length / called_ORF_length` is how much dilution an item imposes: at `f = 1.0` the ORF *is*
the payload, at `f = 0.4` the payload is a minority of what you are handed. It is computable before any
retrieval runs, so it is a **description of the item**, the way a test set says how long its sentences
are. It can never be part of a screening rule, and a method that reads it has stopped doing the task.

## Provenance: where each item came from

Every item says where its sequence came from, so the queries can be traced back rather than taken on
trust.

| column | what it is |
|---|---|
| `source_file` | the record this item was built from, as its tier manifest names it |
| `source_id` | the depositor-facing id, `Addgene #108653`, `ncbi_A13313_1`. **Read from the manifest, never derived from `construct_id`** |
| `source_url` | where to fetch the deposit. Present for the hand tier; the corpus manifest has no URL column, so `source_id` names the record there |
| `filler_source` | `swissprot` for `synth_real`. Empty for `synth_random`, whose filler is random by the tier's definition, and for the deposited tiers, which have no filler |
| `plasmid_id` | `construct_id` before namespacing |
| `orf_strand`, `orf_start`, `orf_end` | 1-based inclusive on the plasmid, the same coordinates `orf_id` packs into a composite key |
| `orf_wraps_origin` | 1 when the ORF crosses the origin of a circular plasmid |

**430 of 1,102 items name a deposit**; the other 672 are the two generated tiers, whose manifests have
no deposit to name. The build prints both counts per tier.

**`orf_wraps_origin` is the one that will bite you.** These plasmids are circular and the ORF caller is
a circular six-frame scanner, so an ORF can start near the end of the sequence and finish past position
1. For those, `end < start`, and `end - start + 1` is **negative**: it is **5 items** here, one in
`hand`, four in `corpus`, and `corpus:ncbi_AF218363_1` would come out at -8,001 nt with nothing
raised. **Take a length from `orf_len`**, which is derived from the translated sequence and is right
for every item. The flag exists to stop a second length being derived, not to fix the first.

**Why the ids are not parsed for this.** `ncbi_M88535_1` -> `M88535` is a rule, and a rule applied to
410 ids is a rule that can be wrong on one of them with nothing in any output to say so. The manifests
already carried the answer; these columns copy it.

## Submitting

Two accepted forms, scored by the same code.

**A ranked list** (`.csv`), columns `construct_id,rank,accession`, any depth. **`construct_id` is
the namespaced id from `items.csv`, `tier:construct_id`**, a submission keyed on the bare id scores
zero everywhere and says so, reporting every item under `missing_from_submission`:

```csv
construct_id,rank,accession
corpus:ncbi_A13313_1,1,B0BCM3
corpus:ncbi_A13313_1,2,A0A017SE81
corpus:ncbi_A13313_1,3,A0A017SQ41
```

**A score matrix** (`.npy`), one row per item in `items.csv` order, one column per entry in
`catalogue_accessions.txt` order, higher is better. This is the only form in which a mean rank is fully
defined, because a true rank deeper than a submitted list is **unknown, not large**. The scorer checks the
shape: a transposed matrix is the one mistake here that would still produce plausible-looking numbers.

```bash
python scripts/53_score_submission.py --benchmark benchmark/v1 --submission my_ranks.csv --arm alignment
python scripts/53_score_submission.py --benchmark benchmark/v1 --submission scores.npy --arm embedding
```

`--arm` is not cosmetic; see the rules below.

## The rules, and why each one exists

All of these are in `src/pilot/benchmark.py`, which is the authority. They are decisions already made,
mostly by me, and mostly after a measurement went wrong.

1. **Every table is stratified by `f`, and there is no pooled scalar across strata.** Not a convention,
   `pilot.benchmark.pooled_scalar` raises. Pooled `recall@10` at the 25% holdout moves +1.0% while the
   `f < 0.50` stratum moves +40.0, because 247 of 286 items are already saturated at depth 10. The pooled
   number is not imprecise, it points the other way.
2. **Mean rank and MRR are primary; `recall@10` is the operational secondary; top-1 is a column and never
   a headline.** Top-1 is blind to the two largest gains in the corpus (rank 76 -> 2 and 79 -> 7, both
   scoring zero) and credits a four-residue trim that broke a near-tie.
3. **For an embedding arm, `recall@10` and MRR are primary and a mean rank may be quoted only beside its
   count of rank >= 10,000 failures.** Under mean-pooling the rank distribution is bimodal where
   alignment's is not: a corpus mean of 1,088 is median rank 1 plus fifteen items carrying one payload at
   ~29,699 of 40,750. `mean_rank_phrase` refuses to render one without the count, which is why `--arm`
   matters.
4. **Micro and macro are both always reported.** Items are not independent trials: 410 corpus constructs
   carry 138 distinct payloads and one appears in 37 of them, so a micro average can be 37 repetitions of
   one answer.
5. **An item that cannot be scored stays in the denominator, as a miss.** Dropping it shrinks `n` exactly
   where the measurement gets hard.
6. **The tiers are never pooled into one number.** They cover each other's weaknesses; a pooled
   figure is a measurement of the mixing proportions.
7. **Review depth 10 is a stated assumption, not a result.** Depths 10 and 25 return the identical winner
   set, so nothing in the data discriminates inside that interval. `--depth` is yours to change; say so if
   you do.

Binding counts are printed before the table, every run. If a limit binds, a submitted list too shallow,
an item missing, an accession not in the catalogue, the count appears, because a table read without them
can be a table about the submission rather than about the method.

## The tiers

Each covers the others' weakness. The f distribution is the reason the first three exist; the
filler is the reason the fourth does.

| tier | items | f < 0.50 | 0.50-0.70 | 0.70-0.95 | f >= 0.95 | what it is for |
|---|---|---|---|---|---|---|
| `hand` | 20 | 4 | 1 | 7 | 8 | ground truth is the depositor's stated insert. Realism, and the anchor |
| `corpus` | 410 | 6 | 15 | 50 | 339 | breadth and statistical power, by gated alignment, 80% validated |
| `synth_real` | 336 | 144 | 72 | 72 | 48 | exact boundaries, SwissProt filler; built **inside** the diluted regime |
| `synth_random` | 336 | 144 | 72 | 72 | 48 | the same 336 constructs with random filler, matched residue for residue on f |

Read the right-hand columns before drawing a conclusion. **In the corpus the ORF usually is the payload**,
339 of 410 items sit at `f >= 0.95`, so a corpus aggregate is largely a measurement of the sampling
frame, and a dilution effect cannot show up in it however real the effect is. That is why `synth_real`
exists, and why rule 1 is a refusal.

**`synth_real` and `synth_random` are the same 336 constructs and differ only in the filler residues**:
same payload accessions, same `orf_aa`, so `f` is identical construct for construct and dilution is
matched. They are two tiers rather than one because the filler **reverses which scorer is hurt**:
random filler is invisible to a local alignment, and costs a mean-pooled embedding arm `recall@10`
0.0% at a median rank of 19,170 against a chance median of 20,375. A benchmark shipping only
`synth_real` poses the worst case for alignment and not for an embedding arm. **Report both, or say
which one a number came from.**

Because the two share every `construct_id`, **item ids in this build are namespaced as
`tier:construct_id`**, for example `synth_real:sc_0001` and `synth_random:sc_0001`. A submission
must return rankings under those ids; `items.csv`, the `queries.faa` headers and the baseline all use
them, and `MANIFEST.json` records the convention as `construct_ids_namespaced`.

## The catalogue

40,750 protein entries, 20,806,731 residues inside the 50-5,000 aa window, built with exact
Smith-Waterman over Swiss-Prot. The search itself is not shipped, because the catalogue is
distributed as the accession list and checksum below rather than rebuilt.

**How it is distributed** (decided 20 Oct): the repository carries the accession list in catalogue order
and a sha256 of the FASTA; **the FASTA itself is attached to the release.** Get it there, then check it:

```bash
python scripts/52_build_benchmark.py --out-dir benchmark/v1 --verify-catalogue path/to/catalogue.fasta
```

That compares the sha256, the entry count, the in-window entry count and the in-window residue total
against `MANIFEST.json`, and exits non-zero if any differ. The checksum is in the repository so that it
can be used, not so that it can be quoted.

The residue count is load-bearing and not trivia: the significance floor is a function of it
(`E = K m N exp(-lambda S)` inverted), so **a different catalogue is a different benchmark**, not a
noisier one. This project has used eighteen reference sets and keeps a register of them, because a
denominator has been carried silently between two of them more than once. The one this
benchmark uses is pinned below by accession list and sha256, which is what stops a reader
repeating that mistake.

**And a rebuild from Swiss-Prot will probably not reproduce the checksum.** The manifests, the builder and
Swiss-Prot itself are all public, so the chain is reproducible in principle. But the **Swiss-Prot release
label was never recorded** when the file was downloaded, around 19 October 2026, only the file's own
sha256 (`a9c3496a...`, 93,801,562 bytes) is pinned, in `MANIFEST.json` under `catalogue.source`. A later
release adds and revises entries, which changes the catalogue, the residue count and therefore the floor.
So the attached FASTA is the authority, a rebuild is a close approximation whose differences are real, and
`--verify-catalogue` is how you tell which one you have. That gap in provenance is ours, and it is the
reason the release carries the data rather than only the recipe.

## The baseline

`benchmark/v1/baseline_alignment.json` and `.csv` hold this project's own arms under these rules, so a
method has something to be compared with. 576 rows: six arms, six holdouts, four strata, four tiers.

It needed **no run**. All 1,102 items already had exact Smith-Waterman ranks on disk, at every
threshold, measured against the same catalogue this directory publishes, 20,806,731 in-window
residues in `MANIFEST.json` and in all four source runs' logs. Reading them costs **1.4 s**;
regenerating them would cost **7 h 17 m** (hand 153.3 s, corpus 2,671.2 s, synth_real 6,769.0 s,
synth_random 16,643.9 s).

**One caveat on those four seconds figures.** The first three were measured on a 16-logical-processor
machine at 14 threads and the fourth on a laptop at 3; the total is therefore a sum over two machines
and is not a prediction of what any one machine would take. A runtime is a measurement and carries
its conditions like any other.

`recall@10` at the 25% holdout, the remote end of the sweep:

| tier | arm | f < 0.50 | 0.50-0.70 | 0.70-0.95 | f >= 0.95 |
|---|---|---|---|---|---|
| synth_real | `whole` | **0.056** | 0.139 | 0.472 | 0.667 |
| synth_real | `bhs_per_segment` | **0.667** | 0.681 | 0.667 | 0.667 |
| synth_real | `oracle` (ceiling, uses the label) | 0.667 | 0.667 | 0.667 | 0.667 |
| synth_random | `whole` = `bhs_per_segment` = `oracle` | 0.667 | 0.667 | 0.667 | 0.667 |
| corpus | `whole` | 0.500 | 0.533 | 0.480 | 0.726 |
| corpus | `bhs_per_segment` | 0.833 | 0.600 | 0.480 | 0.726 |
| hand | `whole` | 0.000 | 1.000 (n=1) | 0.571 | 0.875 |
| hand | `bhs_per_segment` | 0.250 | 1.000 (n=1) | 0.714 | 0.875 |

Read the `synth_real` block, which is the tier built inside the diluted regime. `oracle` is flat at 0.667
by construction, it queries with the payload span, so dilution cannot touch it, and 0.667 is just the 16
of 24 payloads whose family is reachable at all at this holdout. `whole` climbs 0.056 -> 0.667 as dilution
disappears, meeting the ceiling only where there is no dilution left. And `bhs_per_segment`, which is
**label-free**, sits on the ceiling in every stratum.

That is the thesis: the decomposition step recovers what the label would have given you, at every
dilution level, without being told the answer. The corpus rows show its boundary in the same table, at
`f >= 0.95` the two arms are identical to three decimals, because there is nothing to decompose.

Two honest limits on that reading. The ceiling is payload-limited at 16 of 24, so the resolution of the
`synth_real` column is one payload, about 0.042, the 0.681 is one construct, not a trend. And
`synth_real` is a constructed condition, so it is a bound on the real case rather than a measurement of
it; the corpus and hand rows are the measured ends.

**The `synth_random` row is flat at 0.667 everywhere, and that is the design rather than a bug.** Every
arm, every stratum, one value. Two measured facts meet there: random filler is invisible to a local
alignment, so `whole` is bit-identical to `oracle` on all 336 constructs, and the tier is
payload-balanced, so each stratum sees the same 16-of-24 reachable payloads. It is what a stratum
assignment silently failing to apply would also look like, which is why it is said here. **Under a
mean-pooled embedding arm the same tier is the worst case in the benchmark**, `recall@10` 0.0% at a
median rank of 19,170 against a chance median of 20,375, so the flatness is a statement about
alignment, not about the tier.

### Erratum, 20 Oct: the superseded baseline's `bhs` rows overstate decomposition's cost

**This describes `baseline_*_prer10.json`, not the canonical pair.** The defect below was found in the
baseline's own numbers after it was first written; it was then fixed in the harness, every cell was
re-measured, and the re-measured pair is what `baseline_alignment.json` and `baseline_embedding.json`
now hold. The pre-fix files are kept under the `_prer10` suffix, with `is_canonical: false` inside
them, because they are this erratum's evidence and deleting them would leave only the prose.

It is recorded here rather than left for a reader to hit, and it is worth reading even though the
numbers it describes are no longer the headline: it is a defect that was invisible in every output,
that inflated a reported **cost** rather than a reported gain, and whose own source comment described
its direction of bias wrongly.

**What it is.** The baseline's two `bhs` arms are segmented by `scripts/10_twilight.py`, which builds the
segmenter's holdout mask **once at the strictest threshold in the sweep** and reuses that one segmentation
at all six. A construct whose payload has *every* relative masked at the strictest threshold therefore has
nothing in the catalogue for its payload half to match, so the segmenter stops after claiming only the
co-encoded gene, and that one segmentation is then scored at the *looser* holdouts too, where the
payload's remote relatives are back in the database and would have been claimed.

**Where it bites, measured.** Two corpus constructs, of the 305 with an emitted segment:

| construct | payload | f | recorded `bhs_per_segment` | `whole` | under its own holdout mask |
|---|---|---|---|---|---|
| `ncbi_M88535_1` (CTAP-III-hirudin fusion) | P01050 | 0.4012 | **13,225** at 0.70, 13,228 at 0.90, 26,362 at 1.00 | 1 | **1** |
| `ncbi_AF087958_1` (pMA632, phoA-lacZ fusion) | P00634 | 0.7815 | **1,470** at 0.35, 0.50, 0.70 | 1 | **1** |

In both cases the segmenter claimed a real co-encoded gene correctly and the answer key names the other
one; neither is a label error, and neither is a defect in the segmenter. `ncbi_M88535_1` alone is the whole
of the corpus `f < 0.50` cell at the 0.70 holdout, where `bhs_per_segment` reads mean rank **2205.0**
against `whole` **1.33** on n = 6.

**Which cells to read with this in mind.** Every `bhs` and `bhs_per_segment` row for the **corpus** tier at
thresholds **0.35 and looser**. The quantity reported there is a property of this project's segmentation
protocol, not of decomposition.

**The 25% holdout is unaffected, by construction, not by luck.** At 0.25 the segmentation mask *is* the
run's own, so the reuse cannot bite; both constructs above are in fact unscorable at 0.25 because every
one of their homologs is held out. Every `recall@10` figure in the table above, and the thesis claim it
supports, is measured at 0.25.

**What the corpus rows look like with the segmentation they should have had.** All **400** corpus items
scorable at the 0.70 holdout were re-segmented under that holdout's own mask (44 min, same catalogue, same
floor, same scorer; segmentation only, every other arm read from the finished runs). 21 of 400 change,
16 better and 5 worse, net **-14,711** rank points, and the corrected `bhs_per_segment` mean rank is
**identical to `whole`** in three of the four f strata and a genuine gain in the fourth:

| f stratum | n | `whole` | published `bhs_per_segment` | own mask |
|---|---|---|---|---|
| `f < 0.50` | 6 | 1.33 | **2205.00** | **1.33** |
| `0.50-0.70` | 15 | 2.07 | 1.27 | 1.27 |
| `0.70-0.95` | 48 | 1.10 | **31.73** | **1.10** |
| `f >= 0.95` | 331 | 1.14 | 1.20 | 1.14 |

So the published rows overstate decomposition's cost on this tier at this holdout, and the corrected
answer is that it costs nothing. The five constructs that get *worse* under correct segmentation cost +6
rank points between them, and in every one the corrected arm merely equals `whole`, the reused
segmentation had manufactured a one- or two-place win by discarding residues that were competing for the
rank. The reuse is therefore not conservative in either direction, which is a correction to the first
version of this erratum.

Not re-measured: only the 0.70 holdout was re-segmented, alignment only, and the hand and `synth_real`
tiers were not checked at all, `synth_real`'s f distribution is the opposite of the corpus's (64.3% below
0.70 against 5.2%), so no count here transfers to it.

**The harness was fixed on 20 Oct and every cell has since been re-measured.** `10_twilight.py` now
builds the segmenter's mask per threshold, with the payload's own entry masked at threshold 1.0 where
the identity predicate selects nothing; the old behaviour survives behind `--bhs-mask strictest` and
reproduces the `_prer10` files bit-for-bit, which is what makes them a usable record rather than a
stale one. The re-run covered hand, corpus and `synth_real` under both scorers: **9,593.5 s** of exact
alignment and **1,918.5 s** of GPU replay. `whole` and `oracle` moved on **0 of 144** cells, which is
the control proving a segmentation defect cannot reach them; the four segmented arms moved on 50 of
288 under alignment (50 up, 0 down) and 80 of 288 under the embedding arm (**71 up, 9 down**). Those
nine are why *at least the old numbers were a lower bound* was an inference and not a fact.
The segmentation trace, the probe conditions, the binding counts, the full 0.70 re-segmentation
and the hand-tier validation of the fix are recorded in the working repository. What a user of
this benchmark needs is above, and the superseded pair ships here as `baseline_*_prer10` so the
move is checkable from the release alone.

### The baseline will not match this project's own internal summaries

Same ranks, different denominator. This benchmark keeps an unscorable item in the denominator as a miss
(rule 5); the twilight summaries drop it and report over the scorable subset. Across the 64 headline
cells the gap is **mean +0.123 `recall@10`, maximum +0.520, exactly equal in 29 of 64**, always in the
same direction. `baseline_alignment.json` carries the per-cell table under `denominator_difference`.

Neither rule is wrong. This one is the stricter, and it is the one a benchmark should use: an item nobody
can retrieve is a result.

### The two scoring routes are checked, not assumed

The claim at the top of this file, that a reviewer's score and this project's differ by method and
nothing else, has a half that unit tests cannot reach: that `pilot.benchmark`'s holdout construction and
ranking reproduce what `scripts/10_twilight.py` does. Those are different code paths.

```bash
python scripts/55_crosscheck_benchmark.py --tier hand --threads 4
```

Scores the published `queries.faa` for the hand tier against the catalogue, ranks it through the
benchmark, and compares **every cell** against the measurement script's own output. Result:
**120 of 120 cells identical**, 54 s, on the four-tier build published here. Cell by cell rather than
on aggregate, because an aggregate can agree while the cells do not, a rank up in one construct and
down in another leaves `recall@10` untouched. There is no tolerance parameter and the script exits
non-zero on a single mismatch.

**It also exits non-zero when it compares nothing**, which is a different guarantee and was added on
22 Oct after the script was caught passing vacuously. Namespaced item ids made every lookup miss, so
it compared 0 of 120 cells, counted all 120 as absent from the published output, and still printed
that every compared cell was identical, because the success branch tested *no mismatches* and zero
cells produce zero mismatches. The check standing behind the claim at the top of this file had become
a check of nothing. Both halves are fixed and both are pinned by tests.

## What this benchmark does not establish

- **It is a retrieval benchmark, not a screening system.** Ranks move with catalogue, holdout, stratum and
  scorer. "Found" here means a genuine homolog of the payload reached a given rank once its close
  relatives were removed; it does not mean a biosecurity screen would have flagged anything.
- **The hand tier cannot produce every failure.** It contains no item where the two arms differ at
  `k = 1`, which is how a refuted rule once certified itself. Name the falsifying case before trusting a
  subset of these items.
- **The corpus answer key is alignment-derived**, gated at `E <= 1e-3` and validated against the 23
  hand-audited deposits at 16/20. It is good evidence, not a depositor's word.
- **`synth_real` is a constructed condition.** Where it overshoots reality it is reported as a bound, and
  the real case is bracketed between conditions rather than fitted to match.
- **The baseline's own `bhs` rows are not a clean measurement at every holdout.** At 0.35 and looser they
  carry this project's segmentation protocol as well as its arm, see the 20 Oct erratum above. A
  submission's `bhs`-style arm is scored by the rules in `pilot.benchmark` and inherits none of this; the
  caveat is on the comparison row, not on the task.
- **A single-answer key under-reports a multi-gene construct.** `items.csv` names one payload per item, so
  an arm that correctly identifies a construct's *other* real gene scores zero. `ncbi_M88535_1` is the
  worked example: a CTAP-III-hirudin fusion whose depositor annotates both halves.

## Reproducing the build

```bash
python scripts/52_build_benchmark.py --out-dir benchmark/v1
python -m pytest tests/test_benchmark.py
```

The build derives `f` by calling `pilot.arms.build_arms`, the same function the project's own measurements
use, so there is exactly one definition of `f` in the repository. There used to be two, and the second one
read the longest ORF rather than the payload-bearing one, which produced a wrong hand-tier distribution
that the tier totals could not have caught.

## The presentation

`benchmark/presentation.pptx` is this project's own slide deck on the problem the benchmark is cut
from: why a mean-pooled embedding of a whole reading frame loses a payload that a local alignment
keeps, what decomposition recovers, and what it does not. It is here because the rest of this
directory tells you the task, the rules and the answer key and says nothing about why the task has
this shape, and that account already existed.

Three things to know about it.

**It is written by the deck build, not copied here by hand.** A second copy of a binary is a file
that silently stops matching the thing that generates it, which is the defect this project fixes in
both halves every time it finds one, so the build writes this path directly.

**It is not covered by `MANIFEST.json`.** Those checksums pin the data a submission is scored
against. A document is not that, and a checksum over it would imply a guarantee about the benchmark
that it does not carry.

**It is checked, in the one way a packaged document can be.** None of the text rules that run over
this repository on the way out reach inside a compressed archive of markup, so the release build
reads the deck's own text runs instead and refuses to publish on a single pointer to something the
reader does not have: 1,606 runs and 126,000 characters, none of them such a pointer. The deck is
monochrome for everything it draws itself, and the measurement figures it embeds carry the colour
their own scripts emitted.

## Licence

**This repository's own contents are MIT**: the library, the scripts, and the benchmark's items,
homolog table and baselines, all of which are this project's own derivation. The full text is in
`LICENSE` at the repository root.

**`queries.faa` is not free of third-party sequence, and counting was the only way to find out.**
The payload of every item is a real Swiss-Prot protein, and **859 of the 1,102 queries contain
their payload's catalogue entry verbatim**: 2 of 20 in `hand`, 185 of 410 in `corpus`, and **336
of 336 in each generated tier**. The generated tiers are the ones where it is always true, because
those constructs are built around an exact payload span, which is the opposite of what the word
*synthetic* suggests and the opposite of what this file said before the count was run. So UniProt's
CC BY 4.0 attribution travels with the queries as well as with the catalogue. Per-tier counts are
in `MANIFEST.json` under `tiers[].queries_containing_payload_verbatim`.

**The catalogue is not.** `reference_twilight.fasta` is derived from UniProt Swiss-Prot, which is
distributed under **CC BY 4.0**, and that attribution travels with the file: it is attached to the
release rather than committed here, and `catalogue_accessions.txt` plus the sha256 in `MANIFEST.json`
are what the repository carries. Cite UniProt if you use it.

The distinction matters because the two are mixed in every measurement and separable in every file. A
score computed here is this project's work over UniProt's data; redistributing the data carries
UniProt's terms, and redistributing the score does not.

`CITATION.cff` at the repository root is the citation metadata, and `MANIFEST.json` carries the same
split under `licence` so a tool reading the benchmark does not have to parse prose.

## Scope

Benign sequence only: reporters, common tags, standard enzymes, CRISPR tools, standard vectors. No
pathogen- or toxin-derived sequence is fetched, built or analysed here, and nothing in this repository
constructs sequences designed to evade screening. Red-teaming is permanently out of scope;
`SECURITY.md` in this repository gives the three reasons and the benign robustness work that
replaces it.
