"""The benchmark's rules, as code rather than as prose in this project's working rules.

Everything in this module is a decision already made and recorded, mostly by the user, and mostly because
an earlier version of the measurement got it wrong. Writing them out as functions does three things that
restating them in a README does not: a reviewer's own run cannot quietly use a different rule, the rules
become testable, and the two that are *refusals* can actually refuse.

The rules, with where they come from:

* **Payload fraction `f = oracle_span_length / called_ORF_length` strata 0.50 / 0.70 / 0.95**, and
  **every table is stratified by f** (user, 17 Oct). `f` is computable before any
  retrieval runs, so stratifying by it describes the benchmark item; it uses the label and can therefore
  **never** be part of a screening rule. `stratum_of` is the only spelling of the bands.
* **No pooled scalar across strata, ever.** `pooled_scalar` exists only to raise. Pooled `recall@10` at
  the 25% holdout is +1.0% while the `f < 0.50` stratum is +40.0, because 247 of 286 constructs are
  already saturated at depth 10; there is no scalar at any depth that carries the result pooled.
* **Mean rank and MRR are primary, `recall@10` is the operational secondary, top-1 is a column and
  never a headline** (26.3). Depth 10 was fixed by the user 19 Oct (42.1) and is a stated assumption
  about review practice, not a measurement: depths 10 and 25 return the identical winner set.
* **For an embedding arm, `recall@10` and MRR are primary and a mean rank may be quoted only beside its
  count of rank >= 10,000 failures** (32.2, user, 18 Oct). Under mean-pooling the rank distribution is
  bimodal where alignment's is not, so the mean stops describing the centre. `mean_rank_phrase` is the
  only sanctioned rendering, and for an embedding arm it refuses to render without the count.
* **Micro and macro are both always reported.** Constructs are not independent trials: 410 corpus
  constructs carry 138 distinct payloads and one appears in 37 of them, so a micro average can be 37
  repetitions of one answer (17.2, 26.3).
* **An unscorable item stays in the denominator.** A construct whose best surviving target cannot be
  ranked is a miss, not an absence. Dropping it shrinks `n` with difficulty and flatters every curve.
* **Report the best identity a holdout leaves, per payload, beside the holdout**. A
  threshold is an upper bound on what is available, not a description of it: on `synth_real` the 0.70
  holdout leaves 21 of 24 payloads a relative at 0.47 to 0.70 and leaves `P09751` one at **0.268**, a
  25% holdout wearing a 70% label, and that one payload is the embedding arm's only failure in the
  tier at that holdout. `best_identity_left` is the quantity and `holdout_phrase` the only sanctioned
  rendering; it takes the quantity as a required argument, so a bare nominal threshold cannot be
  rendered through it at all.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .align import best_target_rank
from .homology import holdout_mask, targets_at

#: The holdout sweep. Identity above the threshold is removed from the catalogue; what remains at or
#: below it is what counts as a correct answer. 1.00 is the no-holdout control, where the payload's own
#: entry is the answer, it measures string matching and is kept only as the top of the curve.
THRESHOLDS: tuple[float, ...] = (1.00, 0.90, 0.70, 0.50, 0.35, 0.25)

#: Declared in advance, lower bound inclusive and upper exclusive, so every `f` lands in exactly one.
#: The same bands as `scripts/29_stratify_by_payload_fraction.py` and `catalogue_dilution.F_STRATA`; a
#: test pins the agreement, because three spellings of one set of boundaries is how they drift.
STRATA: tuple[tuple[str, float, float], ...] = (
    ("f < 0.50", 0.0, 0.50),
    ("0.50-0.70", 0.50, 0.70),
    ("0.70-0.95", 0.70, 0.95),
    ("f >= 0.95", 0.95, 1.0000001),
)

#: Review depth for `recall@k`. Fixed by the user 19 Oct, not swept here.
DEPTH = 10

#: The rank at or above which an embedding arm's failures are counted separately.
FAILURE_RANK = 10_000


class PooledScalarRefused(Exception):
    """Raised instead of returning a number that the working record forbids."""


def stratum_of(f: float) -> str:
    """The one stratum `f` belongs to. An `f` outside every band is a bug, not a category."""
    for name, lo, hi in STRATA:
        if lo <= f < hi:
            return name
    raise ValueError(f"f={f!r} falls in no stratum; oracle_len > orf_len should be impossible")


def pooled_scalar(*_args, **_kwargs):
    """Never returns. The one rule in this file that has to be able to refuse.

    Pooling across f strata is the mistake the project made before 17 Oct, and the reason it is a
    refusal rather than a warning is that the pooled number is not merely imprecise: it points the
    other way. At the 25% holdout pooled `recall@10` moves +1.0% while the `f < 0.50` stratum moves
    +40.0, because 247 of 286 constructs are saturated at depth 10 and cannot show the effect.
    """
    raise PooledScalarRefused(
        "a pooled scalar across f strata is not reportable (user, 17 Oct). Report the "
        "stratum rows from `report()`; if one number is wanted, name the stratum it belongs to.")


# --------------------------------------------------------------------------------------- items

@dataclass(frozen=True)
class Item:
    """One benchmark item: a construct, the protein a method is handed, and the answer.

    `f` is carried rather than recomputed so that a submission is scored against the same stratum
    assignment the item table was published with. It is derived from `len(arms.orf_seq)`, the
    payload-bearing ORF, and **not** from the longest ORF: that mistake produced a wrong hand-tier f
    distribution on 17 Oct that the tier totals could not have caught.
    """
    construct_id: str
    tier: str
    payload_accession: str
    orf_id: str
    orf_len: int
    oracle_len: int

    @property
    def f(self) -> float:
        return self.oracle_len / self.orf_len if self.orf_len else 0.0

    @property
    def stratum(self) -> str:
        return stratum_of(self.f)


def load_items(path: Path) -> list[Item]:
    """Read `items.csv` as the build script wrote it."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return [Item(r["construct_id"], r["tier"], r["payload_accession"], r["orf_id"],
                     int(r["orf_len"]), int(r["oracle_len"])) for r in csv.DictReader(fh)]


# ------------------------------------------------------------------------------------- metrics

def metrics(ranks: list[int | None], *, depth: int = DEPTH) -> dict:
    """The metric row for one (arm, threshold, stratum) cell.

    `ranks` holds one entry per item in the cell, `None` where the item could not be scored. **Those
    stay in the denominator**: a construct whose best surviving target cannot be ranked is a miss, and
    dropping it would shrink `n` exactly where the measurement gets hard.
    """
    n = len(ranks)
    got = [r for r in ranks if r is not None]
    return {
        "n": n,
        "n_unscorable": n - len(got),
        f"recall_at_{depth}": round(sum(1 for r in got if r <= depth) / n, 4) if n else None,
        "mrr": round(sum(1.0 / r for r in got) / n, 4) if n else None,
        "mean_rank": round(sum(got) / len(got), 2) if got else None,
        "median_rank": int(np.median(got)) if got else None,
        "n_rank_ge_failure": sum(1 for r in got if r >= FAILURE_RANK),
        "top1": sum(1 for r in got if r == 1),
    }


def macro_by_payload(pairs: list[tuple[str, int | None]], fn, nd: int = 4) -> float | None:
    """`fn` applied within each payload, then averaged over payloads, each payload counting once.

    Pass the same `(payload, rank)` pairs the micro figure was computed from, so the two differ only in
    the averaging. `fn` receives the list of that payload's ranks, `None` included.
    """
    per: dict[str, list[int | None]] = {}
    for acc, r in pairs:
        per.setdefault(acc, []).append(r)
    if not per:
        return None
    return round(sum(fn(v) for v in per.values()) / len(per), nd)


def mean_rank_phrase(row: dict, *, arm_kind: str) -> str:
    """The only sanctioned way to render a mean rank.

    For an embedding arm the working record permits a mean **only beside its count of rank >= 10,000
    failures**, so this refuses to render one without the count rather than leaving the caveat to prose.
    The corpus `whole` mean of 1,088 at the 90% holdout is median rank 1 plus 15 constructs carrying one
    payload at ~29,699 of 40,750; printed bare it describes nothing.
    """
    if arm_kind not in ("alignment", "embedding"):
        raise ValueError("arm_kind must be 'alignment' or 'embedding'")
    if row.get("mean_rank") is None:
        return "no scorable item"
    if arm_kind == "alignment":
        return f"{row['mean_rank']:.2f}"
    if "n_rank_ge_failure" not in row:
        raise ValueError("an embedding arm's mean rank may not be quoted without its "
                         f"rank >= {FAILURE_RANK} count")
    return (f"{row['mean_rank']:.2f} ({row['n_rank_ge_failure']} of {row['n']} at rank "
            f">= {FAILURE_RANK:,})")


def best_identity_left(identities, threshold: float) -> float | None:
    """The best identity one payload's holdout actually leaves. `None` when it leaves nothing.

    **A holdout threshold is an upper bound on what is available, not a description of it**
. "Holdout 0.70" means nothing above 0.70 survives; it does not mean that something
    near 0.70 does, and which of the two it means is a per-payload property of the family's identity
    distribution that no output here printed before 21 Oct. `P09751`'s family has a hole between 0.871
    and 0.268, so the 0.70 holdout hands it a relative at 0.268, and a row read as "the embedding arm
    loses 14 constructs to dilution at 0.70" is then two mistakes at once: it is one payload, and it is
    remoteness rather than dilution.

    `identities` are the identities of the payload's homologs **that are in the catalogue**, since a
    relative the catalogue does not hold is not available at any threshold. At `threshold >= 1.0` there
    is no holdout and the payload's own entry is the answer, so the value is its self-identity, 1.0,
    whatever the family looks like, which is why that cell of the sweep measures string matching.
    """
    if threshold >= 1.0:
        return 1.0
    left = [float(i) for i in identities if float(i) <= threshold]
    return max(left) if left else None


def holdout_phrase(threshold: float, best_identity: float | None) -> str:
    """The only sanctioned way to render a holdout threshold for one payload.

    `best_identity` is a required argument rather than an optional one, which is the whole point: the
    bare nominal threshold reads as a description of the regime and is not one, so this function cannot
    be called without the quantity that makes it one. Pass `best_identity_left(...)`, including when
    that is `None`. The same move as `mean_rank_phrase`, where an embedding arm's mean rank cannot be
    rendered without its failure count.
    """
    if best_identity is None:
        return f"holdout {threshold:.2f} (leaves nothing in the catalogue: unscorable)"
    best = float(best_identity)
    if not 0.0 <= best <= 1.0:
        raise ValueError(f"best_identity must be an identity in [0, 1], got {best_identity!r}")
    if best > threshold + 1e-9:
        raise ValueError(
            f"best_identity {best:.4f} is above the holdout {threshold:.2f}, so the targets and the "
            "mask are not the same gated file split at one threshold ")
    return (f"holdout {threshold:.2f} (best relative left {best:.3f}, "
            f"{threshold - best:.3f} below the label)")


def holdout_label_overstated(threshold: float, best_identity: float | None,
                             thresholds: tuple[float, ...] = THRESHOLDS) -> bool:
    """Is this payload's holdout label off by at least one step of the project's own sweep?

    Derived rather than chosen, which is why there is no tunable here: the label overstates when what
    the holdout leaves would also have survived the **next threshold down** in `thresholds`. At 0.70,
    `P09751`'s 0.268 would have survived 0.50 and 0.35 as well, so that cell belongs with the remote
    rows and not with the other 21 payloads at the same nominal threshold. A payload at the bottom of
    the sweep has no next step and is never flagged.
    """
    if best_identity is None or threshold >= 1.0:
        return False
    below = [t for t in thresholds if t < threshold]
    return bool(below) and float(best_identity) <= max(below)


def report(records: list[tuple[Item, float, int | None]], *, depth: int = DEPTH,
           arm_kind: str = "alignment") -> list[dict]:
    """Stratified rows, micro and macro, one per (threshold, stratum). Never a pooled row.

    `records` is one `(item, threshold, rank)` per scored cell, `rank` `None` for an item the method
    could not place. The output carries `arm_kind` so that whatever prints it knows whether a mean rank
    is quotable on its own, and `primary_metrics` so a reader cannot pick the headline themselves.
    """
    if arm_kind not in ("alignment", "embedding"):
        raise ValueError("arm_kind must be 'alignment' or 'embedding'")
    cells: dict[tuple[float, str], list[tuple[Item, int | None]]] = {}
    for item, thr, rank in records:
        cells.setdefault((thr, item.stratum), []).append((item, rank))
    primary = (("mean_rank", "mrr") if arm_kind == "alignment"
               else (f"recall_at_{depth}", "mrr"))
    out = []
    for thr in THRESHOLDS:
        for name, _lo, _hi in STRATA:
            cell = cells.get((thr, name))
            if not cell:
                continue
            ranks = [r for _, r in cell]
            pairs = [(it.payload_accession, r) for it, r in cell]
            row = {"threshold": thr, "stratum": name, "arm_kind": arm_kind,
                   "primary_metrics": list(primary)}
            row.update(metrics(ranks, depth=depth))
            row["n_payloads"] = len({a for a, _ in pairs})
            row[f"recall_at_{depth}_macro"] = macro_by_payload(
                pairs, lambda v: sum(1 for x in v if x is not None and x <= depth) / len(v))
            row["mrr_macro"] = macro_by_payload(
                pairs, lambda v: sum(1.0 / x for x in v if x is not None) / len(v))
            row["mean_rank_macro"] = macro_by_payload(
                pairs,
                lambda v: (sum(x for x in v if x is not None) / len([x for x in v if x is not None]))
                if any(x is not None for x in v) else 0.0, 2)
            out.append(row)
    return out


# ---------------------------------------------------------------------------------- submissions

@dataclass
class Submission:
    """What a method submits, in whichever of the two accepted forms.

    * a **ranked list** per construct, any depth: cheap to produce, and a true rank deeper than the list
      is unknown rather than large, so the scorer reports how many items that bound and refuses to
      average a mean rank over the rest;
    * a **score matrix**, one row per item and one column per catalogue entry in the published orders:
      exact ranks at every depth, and the only form in which a mean rank is fully defined.

    The two are scored by the same function; nothing about the metrics changes with the form.
    """
    kind: str  # "ranked" | "matrix"
    ranked: dict[str, list[str]] | None = None
    matrix: np.ndarray | None = None
    depth_submitted: int | None = None


def load_submission(path: Path) -> Submission:
    """`.csv` -> ranked lists (`construct_id,rank,accession`); `.npy` -> a score matrix."""
    p = Path(path)
    if p.suffix == ".npy":
        return Submission("matrix", matrix=np.load(p))
    rows: dict[str, list[tuple[int, str]]] = {}
    with open(p, newline="", encoding="utf-8-sig") as fh:
        rdr = csv.DictReader(fh)
        need = {"construct_id", "rank", "accession"}
        if not need <= set(rdr.fieldnames or []):
            raise ValueError(f"a ranked submission needs columns {sorted(need)}, got {rdr.fieldnames}")
        for r in rdr:
            rows.setdefault(r["construct_id"], []).append((int(r["rank"]), r["accession"]))
    ranked = {k: [a for _, a in sorted(v)] for k, v in rows.items()}
    depth = max((len(v) for v in ranked.values()), default=0)
    return Submission("ranked", ranked=ranked, depth_submitted=depth)


def rank_from_ranked_list(ordered: list[str], targets: set[str], masked: set[str]) -> int | None:
    """Rank of the first true target in a submitted list, counting masked entries as absent.

    A masked accession is one the holdout removed from the catalogue, so a method that returns it has
    returned something that does not exist at this threshold: it is skipped and does **not** occupy a
    rank. Returning `None` for a list that contains no target is the same convention as
    `best_target_rank`, unknown, not large, and the scorer counts those rather than guessing.
    """
    pos = 0
    for acc in ordered:
        if acc in masked:
            continue
        pos += 1
        if acc in targets:
            return pos
    return None


def rank_cells(sub: Submission, items: list[Item], homologs: dict[str, list],
               catalogue_accessions: list[str]) -> tuple[list[tuple[Item, float, int | None]], dict]:
    """One `(item, threshold, rank)` per cell, plus binding counts. The ranks, before any averaging.

    Split out from `score_submission` so that a rank can be checked against another route's rank
    without going through the metric layer, which is what `scripts/55_crosscheck_benchmark.py` does
    against `10_twilight.py`'s own output. An aggregate agreeing is much weaker evidence than every
    cell agreeing, and only the cells can distinguish "the same rule" from "the same average".
    """
    idx_of = {a: i for i, a in enumerate(catalogue_accessions)}
    if sub.kind == "matrix":
        want = (len(items), len(catalogue_accessions))
        if sub.matrix is None or sub.matrix.shape != want:
            raise ValueError(
                f"a score matrix must be {want}, one row per item in `items.csv` order, one column "
                f"per entry in `catalogue_accessions.txt` order, got "
                f"{None if sub.matrix is None else sub.matrix.shape}. Row or column order silently "
                f"transposed is the one error here that would still produce plausible numbers.")
    records: list[tuple[Item, float, int | None]] = []
    bind = {"items": len(items), "missing_from_submission": 0, "unknown_accessions": 0,
            "rank_beyond_submitted_depth": 0, "no_target_at_threshold": 0,
            "submitted_depth": sub.depth_submitted}
    for row_i, item in enumerate(items):
        hs = homologs.get(item.payload_accession, [])
        # Counted once per item, not once per threshold: a binding count that multiplies by the size of
        # the holdout sweep is not a count of anything.
        if sub.kind == "ranked":
            ordered_item = (sub.ranked or {}).get(item.construct_id)
            if ordered_item is None:
                bind["missing_from_submission"] += 1
            else:
                bind["unknown_accessions"] += sum(1 for a in ordered_item if a not in idx_of)
        for thr in THRESHOLDS:
            if thr >= 1.0:
                masked, targets = set(), {item.payload_accession}
            else:
                masked = holdout_mask(hs, thr)
                targets = targets_at(hs, thr)
            targets &= set(idx_of)
            if not targets:
                bind["no_target_at_threshold"] += 1
                records.append((item, thr, None))
                continue
            if sub.kind == "matrix":
                m_idx = np.array([idx_of[a] for a in masked if a in idx_of], dtype=int)
                t_idx = np.array([idx_of[a] for a in targets], dtype=int)
                r = best_target_rank(sub.matrix[row_i], t_idx, m_idx)
            else:
                if ordered_item is None:
                    records.append((item, thr, None))
                    continue
                r = rank_from_ranked_list(ordered_item, targets, masked)
                if r is None:
                    bind["rank_beyond_submitted_depth"] += 1
            records.append((item, thr, r))
    return records, bind


def score_submission(sub: Submission, items: list[Item], homologs: dict[str, list],
                     catalogue_accessions: list[str], *, depth: int = DEPTH,
                     arm_kind: str = "alignment") -> tuple[list[dict], dict]:
    """Score either submission form against the published holdouts. Returns (rows, binding counts).

    The holdout is built with `pilot.homology.holdout_mask` and `targets_at`, the same two functions the
    project's own runs use, so a reviewer's score and ours differ by method and by nothing else. That
    claim is checked rather than asserted: `scripts/55_crosscheck_benchmark.py` compares every cell
    against `10_twilight.py`'s own ranks on the hand tier.
    """
    records, bind = rank_cells(sub, items, homologs, catalogue_accessions)
    return report(records, depth=depth, arm_kind=arm_kind), bind
