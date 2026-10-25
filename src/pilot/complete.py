"""Masked-LM completion of a short fragment, and the contamination measurement that has to go with it.

**Why this module is small and why its output is a caveat rather than a result.** The fragment extension
(T7) asks whether a short fragment retrieves its payload better after a protein language
model extends it. The obvious way to run that is contaminated beyond repair in this repository, and the
reason is worth stating once in code rather than three times in prose:

* Every holdout here masks the **retrieval catalogue**: `homologs.csv` split by identity threshold,
  targets and mask being the same gated file. A completion step reads the **weights**.
* ESM-2's training set is UniRef (Lin et al., *Science* 2023) and all 173 payloads are Swiss-Prot
  entries, so the weights hold every payload or a close relative of it. At the 0.25 holdout the
  catalogue holds nothing above 0.25 while the weights hold something at >= 0.50, so the completion step
  is handed exactly what the holdout removed.
* **Switching models does not fix it.** ESM-2, ESM-C, ProtT5, ProstT5, ProGen2 and ESM-3 are trained on
  UniRef, UniParc or BFD, all of which contain Swiss-Prot. A swap buys *scorer-independence*
  (this project's lesson) and not a holdout.
* Neither way out is available: the working record forbids new model development or training, and there are no
  post-training-cutoff payloads here to hold out by date.

So "completed it" and "memorised it" predict the same observation, and a retrieval gain alone cannot
separate them. What *can* be reported honestly is the gain **beside** `identity(completion, truth)`: a
completion at 30% identity to the real continuation that still lifts a remote relative's rank is
inference about the family, and one at 95% is retrieval of a memorised sequence: still operationally
useful, and a different claim. This module measures that identity. It is the gate on the extension, not
a part of it.

The falsifying case, named before the run the claim would not transfer to a genuinely
novel payload, and **nothing here measures that**. Scope any result to known-protein payloads.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

#: The 20 standard amino acids, as the tokenizer orders nothing in particular, used only for the
#: uniform baseline, where the ordering is irrelevant and the count is not.
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"


def identity(a: str, b: str) -> float:
    """Fraction of positions where two equal-length strings agree. 0.0 for an empty comparison."""
    if len(a) != len(b):
        raise ValueError(f"identity needs equal lengths, got {len(a)} and {len(b)}")
    return sum(1 for x, y in zip(a, b) if x == y) / len(a) if a else 0.0


def most_frequent_residue(seqs) -> str:
    """The single residue that a composition-only predictor would always emit."""
    c: Counter = Counter()
    for s in seqs:
        c.update(s)
    return max((r for r in c if r in STANDARD_AA), key=lambda r: c[r])


def composition_baseline(seqs) -> float:
    """Expected identity of the best composition-only predictor: the frequency of the commonest residue.

    This is the number a completion has to beat before it has demonstrated anything at all, and it is
    **not** 1/20. On a Swiss-Prot-derived catalogue L alone is near 0.10, so a completion at 0.12
    identity is noise wearing a decimal point.
    """
    c: Counter = Counter()
    for s in seqs:
        c.update(r for r in s if r in STANDARD_AA)
    tot = sum(c.values())
    return max(c.values()) / tot if tot else 0.0


@dataclass
class Completion:
    """One completion and everything needed to judge it."""
    payload_accession: str
    fragment_len: int
    n_completed: int
    mode: str
    completed: str
    truth: str

    @property
    def identity_to_truth(self) -> float:
        return identity(self.completed, self.truth)


class MaskedLmCompleter:
    """Extends a peptide with an ESM-2 masked-LM head.

    The cached `facebook/esm2_t33_650M_UR50D` checkpoint is already an `EsmForMaskedLM`: 574 tensors
    including `lm_head.dense.weight` and `lm_head.bias`, `architectures: ["EsmForMaskedLM"]` in its
    config. So this needs no download and no new dependency, which is why the extension
    was never compute-bound.

    Two modes, because they answer slightly different questions and the difference is informative:

    * `"iterative"` (default) fills the leftmost mask, appends the winner, and repeats: `n` forward
      passes, and the only mode whose output is a *sequence* the model would assign high likelihood to.
    * `"single"` appends `n` masks and takes the argmax at each in one pass. An MLM's masked positions
      are conditionally independent given the context, so this is cheaper and systematically worse at
      the far end of a long completion. Reported rather than hidden, because a reader comparing a
      completion study to this one needs to know which was used.
    """

    def __init__(self, model_name: str = "facebook/esm2_t33_650M_UR50D", device: str | None = None,
                 max_length: int = 1022):
        self.model_name, self.max_length = model_name, max_length
        self.name = model_name.split("/")[-1]
        self._device, self._tok, self._model = device, None, None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: PLC0415
        from transformers import AutoTokenizer, EsmForMaskedLM  # noqa: PLC0415

        self._torch = torch
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = EsmForMaskedLM.from_pretrained(self.model_name).to(self._device).eval()
        # Only the 20 standard residues may be emitted. Without this the head is free to return `X`,
        # `B`, `<mask>` or `<pad>`, and an identity against truth would then be scored against tokens
        # that are not residues: the coverage-denominator mistake in a new costume.
        self._allowed = self._torch.tensor(
            [self._tok.convert_tokens_to_ids(r) for r in STANDARD_AA], device=self._device)

    def complete(self, prefix: str, n: int, mode: str = "iterative") -> str:
        """`n` residues continuing `prefix`, from the masked-LM head."""
        if mode not in ("iterative", "single"):
            raise ValueError("mode must be 'iterative' or 'single'")
        if n <= 0:
            return ""
        self._load()
        torch, tok, model = self._torch, self._tok, self._model
        if mode == "single":
            text = prefix + tok.mask_token * n
            enc = {k: v.to(self._device) for k, v in tok(text, return_tensors="pt").items()}
            with torch.no_grad():
                logits = model(**enc).logits[0]
            ids = enc["input_ids"][0]
            pos = (ids == tok.mask_token_id).nonzero(as_tuple=True)[0]
            out = []
            for p in pos:
                sub = logits[p][self._allowed]
                out.append(STANDARD_AA[int(sub.argmax())])
            return "".join(out)
        grown = prefix
        for _ in range(n):
            text = grown + tok.mask_token
            enc = {k: v.to(self._device) for k, v in tok(text, return_tensors="pt").items()}
            with torch.no_grad():
                logits = model(**enc).logits[0]
            ids = enc["input_ids"][0]
            p = (ids == tok.mask_token_id).nonzero(as_tuple=True)[0][0]
            sub = logits[p][self._allowed]
            grown += STANDARD_AA[int(sub.argmax())]
        return grown[len(prefix):]


# --------------------------------------------------------------------------------------------------
# R15: turning a completion into a query.
#
# `51_fragment_probe.py` cuts its own fragments and takes one query per row, so nothing in this
# repository could feed it a *completed* fragment. The gap is this: a completion is a string, and the
# question R15 asks is whether appending it to the fragment moves the payload's rank. That needs four
# queries and not one, because three things change at once when a completion is appended, the query
# gets longer, it gains residues of some composition, and it gains residues in a particular order,
# and only the third is the model's contribution (73).
# --------------------------------------------------------------------------------------------------

#: The four arms, in the order they are reported. `fragment` is the control `51_fragment_probe.py`
#: already measures; `truth` is the ceiling, and if it does not move the rank then neither can the
#: model and the run says nothing about the model. Named here rather than in the script so a test can
#: assert the set.
#: `GENERIC_ARMS` is defined at the foot of this module, beside the function that builds them.
EXTENSION_ARMS = ("fragment", "extended", "shuffled", "truth", "homopolymer", "generic",
                  "misconditioned")


def deterministic_shuffle(s: str, key: str) -> str:
    """A composition-preserving permutation of `s`, fixed by `key` and stable across processes.

    Python's `hash()` is salted per process, so a shuffle seeded from it would silently differ between
    the ThinkPad dry run and the Omen run and the two could not be compared. The seed is taken from
    sha256 instead, which is the same everywhere and for all time.
    """
    import hashlib  # noqa: PLC0415
    import random  # noqa: PLC0415

    chars = list(s)
    random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")).shuffle(chars)
    return "".join(chars)


def extension_arms(fragment: str, completed: str, truth: str, key: str,
                   tails: dict[str, str], donor: str) -> dict[str, str]:
    """The seven queries one completion yields, each isolating one thing the extension changes.

    * **`fragment`**: the fragment alone. The control, and the same string `51_fragment_probe.py`
      would have queried, so a rank here is comparable with that script's.
    * **`extended`**: fragment + the model's completion. The arm.
    * **`shuffled`**: fragment + the model's own completion with its residues permuted. Same length,
      same composition, no order. A gain that survives here is a length-and-composition effect and not
      inference, which matters because a longer query reaches a higher raw alignment score for free.
    * **`truth`**: fragment + the real continuation. The **ceiling**: it is the best any completion
      of this length could do. If `truth` does not beat `fragment`, the manipulation is too small to
      move the measurement and no statement about the model is available from the run: that is the
      falsifying case, it is checkable before any GPU time, and it is checkable with no model loaded.

    * **`homopolymer`** and **`generic`**: fragment + a tail that is the **same string for every
      query in the run** (`fixed_tails`, 23 Oct). `shuffled` still carries this query's own residue
      composition; these carry nothing of it at all. They are the control this project named and
      predicted the outcome of: if lengthening a short query with anything protein-shaped is what
      moves a pooled embedding toward its family, these two match `extended` and the model is not
      part of the claim.

    `truth` is the labelled continuation, so like `f` it is a description of the item and can never be
    part of a screening rule.

    * **`misconditioned`**: fragment + **another payload's** completion (`donor_completions`,
      23 Oct). Same model, same length, a real completion of a real protein context: everything
      `extended` has except the fragment it was conditioned on. 77.5's control, and the last
      explanation left standing after `generic` failed.

    `tails` and `donor` are required rather than defaulted. A default would let a caller silently
    get four arms where the summary expects seven, and an absent arm reads as a measured blank:
    the shape of defect this file's own register collects.
    """
    if len(completed) != len(truth):
        raise ValueError(f"completion and truth must be the same length, got {len(completed)}  "
                         f"and {len(truth)}")
    missing = [a for a in GENERIC_ARMS if a not in tails]
    if missing:
        raise ValueError(f"no fixed tail supplied for {missing}; build them with fixed_tails()")
    wrong = {a: len(tails[a]) for a in GENERIC_ARMS if len(tails[a]) != len(completed)}
    if len(donor) != len(completed):
        wrong["misconditioned"] = len(donor)
    if wrong:
        raise ValueError(f"every tail must be {len(completed)} residues, the completion's length; "
                         f"got {wrong}")
    return {
        "fragment": fragment,
        "extended": fragment + completed,
        "shuffled": fragment + deterministic_shuffle(completed, key),
        "truth": fragment + truth,
        "homopolymer": fragment + tails["homopolymer"],
        "generic": fragment + tails["generic"],
        "misconditioned": fragment + donor,
    }


def read_completion_sweep(path, lengths=None, mode: str | None = None) -> list[Completion]:
    """Rows of a `completion_sweep_*.csv` as `Completion`s, optionally filtered.

    The sweep carries no `start` column, so **where each fragment was cut is not recorded** and has to
    be re-derived rather than assumed: the same problem R16's provenance posed. The
    convention is a prefix cut, `fragment = payload[:m]` and `truth = payload[m:m + n]`, and
    `check_prefix_cut` below is what establishes it against the catalogue instead of trusting this
    sentence.
    """
    import csv  # noqa: PLC0415

    want = None if lengths is None else set(lengths)
    out = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if want is not None and int(r["fragment_aa"]) not in want:
                continue
            if mode is not None and r["mode"] != mode:
                continue
            out.append(Completion(
                payload_accession=r["payload_accession"], fragment_len=int(r["fragment_aa"]),
                n_completed=int(r["n_completed"]), mode=r["mode"],
                completed=r["completed"], truth=r["truth"]))
    return out


def check_prefix_cut(rows, seq_of) -> tuple[int, list[str]]:
    """Confirm the prefix-cut convention against the catalogue; return (confirmed, disagreements).

    A disagreement means the recorded `truth` is not the continuation of a prefix cut, so every query
    built from that row would be built on a guess. The caller refuses rather than proceeding: this
    is 69.6's lesson about a check that can pass having compared nothing, so the confirmed count is
    returned and asserted by the caller and not merely printed.
    """
    ok, bad = 0, []
    for c in rows:
        s = seq_of(c.payload_accession)
        if s is None:
            bad.append(f"{c.payload_accession}: not in the catalogue")
            continue
        if s[c.fragment_len:c.fragment_len + c.n_completed] == c.truth:
            ok += 1
        else:
            bad.append(f"{c.payload_accession}@{c.fragment_len}: truth is not the prefix continuation")
    return ok, bad


# ---- the control that would retire the model from this project's claim -------------------------

#: The two query-independent tails, by arm name. Both are the SAME STRING for every query in a run,
#: which is the whole point: they carry nothing about the fragment they are appended to.
GENERIC_ARMS = ("homopolymer", "generic")


def fixed_tails(seqs, n: int, key: str = "plasmid-pilot:fixed-tail") -> dict[str, str]:
    """Two fixed tails of length `n`, derived from a catalogue rather than chosen.

    **Why this exists** (named there as the control nobody had run). Under mean-pooled
    ESM-2 the model's completion appears to work: `extended` goes 120 better against 63 worse and
    becomes indistinguishable from appending the real continuation. Three facts in the same run say
    that is displacement rather than recovery: the gain is flat in identity to the truth, the
    completions that collapse to one repeated residue are the **best** subgroup, and the net sits
    where the query already ranked badly. The mechanism those three imply is that twelve residues are
    27% of a 33 aa query's pooled mean, so a short query is a bad query for a pooled embedding and
    lengthening it with anything protein-shaped moves it toward its family.

    If that is right, a tail carrying **no information about the fragment at all** should work about
    as well as the model's, and the model is then not part of the claim. That is a prediction with a
    cheap test, and this is the test.

    * **`homopolymer`**: the commonest residue repeated. This is exactly what `composition_baseline`
      says a composition-only predictor emits, and 74.4 found the completions that degenerate to it
      are the best-performing subgroup, so this arm makes that subgroup deliberate instead of
      accidental.
    * **`generic`**: a composition-matched draw. Protein-shaped in its residue frequencies and in
      nothing else: no order, no motif, no relation to any payload.

    Both are **derived from `seqs`** and neither is a literal, so the catalogue they came from is a
    property to report rather than a constant to defend.

    The draw is sha256-seeded over a **sorted** residue list, for the reason `deterministic_shuffle`
    is: a `Counter`'s iteration order follows insertion, so an unsorted draw would depend on the order
    the catalogue happened to be read in and would differ between two machines with nothing in either
    output saying so.
    """
    import hashlib  # noqa: PLC0415
    import random  # noqa: PLC0415

    if n <= 0:
        raise ValueError(f"a tail needs a positive length, got {n}")
    freq: Counter = Counter()
    for s in seqs:
        freq.update(r for r in s if r in STANDARD_AA)
    if not freq:
        raise ValueError("no standard residues in the sequences the tails are derived from")
    residues = sorted(freq)
    rng = random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big"))
    return {
        "homopolymer": max(residues, key=lambda r: freq[r]) * n,
        "generic": "".join(rng.choices(residues, weights=[freq[r] for r in residues], k=n)),
    }


def donor_completions(rows, key: str = "plasmid-pilot:donor") -> dict[tuple, str]:
    """Each row's completion reassigned to a different row: 77.5's control, keyed per row.

    **What it isolates.** `generic` (77.2) showed that a tail carrying nothing about the query is
    worse than no tail at all, which killed 74.4's "anything protein-shaped" mechanism. It left one
    explanation standing: that what matters is the tail being conditioned on *this* fragment. This
    arm tests exactly that and nothing else. The donor tail is the **same model, the same length and
    a real completion of a real protein context**: every property `extended` has except that it
    was generated from a different fragment.

    So the three controls now bracket the question. `shuffled` keeps this query's composition and
    destroys order. `generic` keeps nothing of the query. `misconditioned` keeps everything except
    which query it was conditioned on.

    **Stated before the run**: 77.5 predicts `misconditioned` lands below `extended`. If
    instead it matches, then conditioning is not what the completion buys and the claim becomes
    "any model-generated peptide of this length helps", which is a different and weaker statement
    that `generic` has already made harder to believe.

    Donors are drawn **within** a (fragment length, mode) group, so length and decoding mode are held
    fixed and only the conditioning moves. The assignment is a sha256-seeded shuffle followed by a
    rotation, which is a derangement by construction: no row can receive its own completion, and
    rotating a shuffled list cannot produce a fixed point because adjacent entries are distinct.
    Sorted first, for `deterministic_shuffle`'s reason: the input order must not reach the output.
    """
    import hashlib  # noqa: PLC0415
    import random  # noqa: PLC0415

    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r.fragment_len, r.mode), []).append(r)
    out: dict[tuple, str] = {}
    for g, rs in sorted(groups.items()):
        if len(rs) < 2:
            raise ValueError(f"group {g} has {len(rs)} row(s); a donor needs someone to borrow from")
        keys = sorted((r.payload_accession, r.fragment_len, r.mode) for r in rs)
        by_key = {(r.payload_accession, r.fragment_len, r.mode): r for r in rs}
        seed = int.from_bytes(hashlib.sha256(f"{key}:{g}".encode()).digest()[:8], "big")
        random.Random(seed).shuffle(keys)
        for i, k in enumerate(keys):
            out[k] = by_key[keys[(i + 1) % len(keys)]].completed
    return out
