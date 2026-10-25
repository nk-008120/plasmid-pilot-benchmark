# SECURITY.md

Scope and safety posture of `plasmid-pilot`. Last updated 16 October 2026.

## What this project is for

Commercial gene synthesis providers screen incoming DNA orders against databases of sequences
of concern. That screening is largely homology-based, and it is weakest on **complex
constructs**: a single synthetic plasmid can carry a payload gene fused to tags, linkers,
markers, promoters and self-cleaving peptides, and the biologically meaningful unit is buried
inside a much larger sequence.

Biological foundation models could screen by predicted function rather than by sequence
similarity, which would catch relatives that alignment misses. But those models expect clean,
single-protein inputs. This project measures whether a **decomposition step**, splitting a
messy construct into model-ready units before screening, recovers the performance those models
lose on real constructs.

The work is defensive. Its output is a benchmark and a measurement of where screening fails,
intended to make detection better.

## What the project does not do

These are enforced as working rules in this project's working rules and have been followed throughout.

- **No pathogen- or toxin-derived sequences.** The project never fetches, builds or analyses
  them. Every sequence here is benign and well characterised: fluorescent proteins, common
  epitope tags, selection markers, standard enzymes, CRISPR tooling and standard vectors.
- **No evasion research.** The project does not create or study obfuscated, fragmented or
  redesigned variants intended to defeat screening. Red-teaming of screening systems is
  explicitly out of scope and is named in the proposal as an extension that would require the
  funding programme to provide information-hazard guidance first.
- **No capability to design or construct anything.** The code calls open reading frames,
  aligns proteins, embeds them, and measures retrieval accuracy. It does not design sequences,
  optimise them for any property, or assemble constructs.

## Why decomposition does not weaken screening

A reasonable question is whether splitting constructs into parts could help someone hide a
concerning sequence. It does not, for a structural reason: decomposition is applied **by the
screener, to the incoming order**, not by the submitter. It increases what a screen can see.
The measured failure mode this project documents is that an undecomposed query *hides*
co-encoded genes, so the status quo is the permissive case and decomposition is the stricter
one.

The benchmark also only ever measures retrieval of benign reference proteins from a benign
catalogue. Nothing in the repository is specific to any sequence of concern, and the method
is agnostic to what is being looked for.

## Data provenance

- 23 hand-audited plasmids from Addgene, chosen for being benign and well documented.
- 789 screened NCBI records, filtered to 410 scorable constructs.
- Swiss-Prot (UniProt reviewed), used as the protein reference and decoy source.
- Downloaded GenBank and FASTA records are treated as untrusted data: parsed, never executed.

## If scope ever changes

Any proposal to work with sequences of concern, to study evasion, or to publish a method for
bypassing a screen is out of scope for this repository. It would need the funding programme
(IRI) and an appropriate biosecurity review to agree the handling first, and it should happen
in a separate, access-controlled project rather than here.

## Contact

Nishit Kalani, principal author. Project description by Kevin Flyangolts (IRI).
