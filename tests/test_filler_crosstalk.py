"""Tests for `scripts/56_filler_crosstalk.py`, the experiment that decides the working record.

The script exists to pick between two mechanisms, so the things worth pinning are the ones that would let
it hand back a *confident* mechanism label that the numbers do not support, or a number that looks like
it answers 60.3 and does not.

Seven of those:

* **the verdict rule is a function of the numbers and is fixed here**, so the label cannot be chosen after
  the model has run. Each branch is driven with a synthetic summary that encodes one hypothesis;
* **attribution is separate from magnitude.** Cross-talk that is material under *both* fillers is a real
  effect on per-residue embeddings and still does not explain a *difference between fillers*, which is
  what 60.2 measured. The rule must not collapse the two;
* **the tier is established by md5**, so a run against the wrong `orfs.faa` refuses rather than producing
  numbers labelled with the right tier (four labels name two tiers);
* **the shared-construct invariant is checked, not cited.** Every paired comparison in the script is void
  if the two manifests are not the same constructs, so a disagreement must stop the run;
* **`f` comes from its definition**, never from the `payload_fraction` column, which is a different
  quantity on this manifest;
* **selection is paired and deterministic.** A payload missing one dilution level is dropped from both,
  so the dose-response cannot quietly change `n` between rows, and `--max-payloads` is a prefix of the
  full run rather than a different sample;
* **a payload that runs past `--max-length` is reported as unmeasurable, never slice-fitted**: the
  truncation limit that this project's table says to count rather than assume.
"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name.split(".")[0], ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ct = _load("56_filler_crosstalk.py")


# ---- the arithmetic ---------------------------------------------------------------------------------

def test_corruption_is_the_fraction_of_the_available_range():
    """1.0 is identical, the background cosine is fully unrelated. Without this scale a cosine of 0.95
    is read against an imagined null, which is this project's error."""
    assert ct.corruption(1.0) == pytest.approx(0.0)
    assert ct.corruption(ct.ESM2_BACKGROUND_COSINE) == pytest.approx(1.0)
    assert ct.corruption(0.9) == pytest.approx(0.1 / (1 - ct.ESM2_BACKGROUND_COSINE), rel=1e-6)


def test_residue_cosines_are_positionwise_and_refuse_a_shape_mismatch():
    rng = np.random.default_rng(61)
    a = rng.normal(size=(7, 5)).astype(np.float32)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    assert np.allclose(ct.residue_cosines(a, a), 1.0)
    assert np.allclose(ct.residue_cosines(a, -a), -1.0)
    with pytest.raises(ValueError):
        ct.residue_cosines(a, a[:3])


def test_payload_fraction_comes_from_its_definition_not_the_column():
    """`payload_fraction` in `data/synthetic_real` is a different quantity: its first row is a 705 aa
    payload in a 705 aa ORF reading 0.3588."""
    row = {"payload_aa": "705", "orf_aa": "705", "payload_fraction": "0.3588"}
    assert ct.payload_fraction(row) == pytest.approx(1.0)


# ---- the pre-registered verdict ---------------------------------------------------------------------

def _summary(level, cos_res, cos_pay, cos_whole, filler="random"):
    return dict(level=level, filler=filler, n_constructs=24, n_payloads=24, median_f=0.2,
                cos_res_mean=cos_res, cos_res_corruption=round(ct.corruption(cos_res), 4),
                cos_res_min=cos_res, pooled_cos_payload_rows=cos_pay,
                pooled_payload_rows_corruption=round(ct.corruption(cos_pay), 4),
                pooled_cos_whole_res=cos_whole, pooled_cos_whole_arm=cos_whole,
                pooled_whole_corruption=round(ct.corruption(cos_whole), 4),
                crosstalk_share_of_deficit=round((1 - cos_pay) / (1 - cos_whole), 4))


def _paired(n=24, lower=24):
    return dict(n=n, random_lower=lower, random_higher=n - lower, equal=0, median_delta=-0.1)


def test_verdict_displacement():
    """The payload's own rows survive intact and only the pooled mean has moved: 60.3's second branch."""
    s = [_summary("0.8", cos_res=0.999, cos_pay=0.998, cos_whole=0.60)]
    v = ct.verdict(s, _paired(lower=12), "0.8")
    assert v["label"] == "displacement"
    assert v["pooled_movement_material"] is True


def test_verdict_crosstalk():
    """The payload's own rows are corrupted, that corruption carries most of the pooled deficit, and it
    is attributable to the random filler specifically."""
    s = [_summary("0.8", cos_res=0.70, cos_pay=0.62, cos_whole=0.60)]
    v = ct.verdict(s, _paired(lower=23), "0.8")
    assert v["label"] == "crosstalk"
    assert v["attributable_to_random_filler"] is True


def test_verdict_both_when_crosstalk_is_material_but_not_dominant():
    s = [_summary("0.8", cos_res=0.85, cos_pay=0.93, cos_whole=0.60)]
    v = ct.verdict(s, _paired(lower=24), "0.8")
    assert v["label"] == "both"


def test_attribution_is_reported_but_is_not_part_of_the_label():
    """A per-residue effect present under BOTH fillers is real and does not explain 60.2's *difference*.
    The mechanism label and the attribution are different questions, so the same cosines must give the
    same label whichever way the paired count falls: with the attribution flag carrying the difference.
    """
    s = [_summary("0.8", cos_res=0.70, cos_pay=0.62, cos_whole=0.60)]
    directional = ct.verdict(s, _paired(lower=23), "0.8")
    flat = ct.verdict(s, _paired(lower=12), "0.8")
    assert directional["label"] == flat["label"] == "crosstalk"
    assert directional["attributable_to_random_filler"] is True
    assert flat["attributable_to_random_filler"] is False


def test_verdict_neither_when_the_pooled_vector_did_not_move():
    """Nothing to explain: 60.2's collapse is not visible in cosine space and neither branch is
    supported. The gate is on the pooled movement, so a high cross-talk *share* of a negligible deficit
    does not promote itself to a mechanism."""
    s = [_summary("0.8", cos_res=0.999, cos_pay=0.995, cos_whole=0.999)]
    v = ct.verdict(s, _paired(lower=12), "0.8")
    assert v["label"] == "neither"
    assert v["pooled_movement_material"] is False


def test_the_magnitude_gate_is_on_the_quantity_being_explained():
    """The regression for the draft rule its own fixture refuted: a pooled deficit of a material 0.12 of
    the range, 71% of it surviving a payload-only mean, is cross-talk, and a draft that gated on the
    per-residue cosine instead called it `neither` because 0.960 sat below the magnitude gate."""
    s = [_summary("0.8", cos_res=0.9604, cos_pay=0.9504, cos_whole=0.9302)]
    assert ct.corruption(0.9604) < ct.VERDICT_MATERIAL          # the per-residue figure is not material
    assert ct.corruption(0.9302) >= ct.VERDICT_MATERIAL         # the pooled movement is
    v = ct.verdict(s, _paired(lower=22), "0.8")
    assert v["label"] == "crosstalk"


def test_verdict_carries_the_rule_it_applied():
    v = ct.verdict([_summary("0.8", 0.999, 0.998, 0.6)], _paired(), "0.8")
    assert v["rule"] == dict(material=ct.VERDICT_MATERIAL,
                             paired_majority=ct.VERDICT_PAIRED_MAJORITY,
                             crosstalk_share=ct.VERDICT_CROSSTALK_SHARE,
                             displacement_share=ct.VERDICT_DISPLACEMENT_SHARE,
                             background_cosine=ct.ESM2_BACKGROUND_COSINE,
                             material_applies_to="corruption(pooled_cos_whole_res) under random filler")


def test_paired_is_a_sign_count_and_needs_no_threshold():
    rows = [
        dict(level="0.8", filler="random", payload_accession="P1", measurable=True, cos_res_mean=0.70),
        dict(level="0.8", filler="real", payload_accession="P1", measurable=True, cos_res_mean=0.95),
        dict(level="0.8", filler="random", payload_accession="P2", measurable=True, cos_res_mean=0.99),
        dict(level="0.8", filler="real", payload_accession="P2", measurable=True, cos_res_mean=0.90),
        # unmeasurable rows must not enter the count in either direction
        dict(level="0.8", filler="random", payload_accession="P3", measurable=False, cos_res_mean=0.0),
        dict(level="0.8", filler="real", payload_accession="P3", measurable=True, cos_res_mean=0.9),
    ]
    p = ct.paired(rows, "0.8", "cos_res_mean")
    assert (p["n"], p["random_lower"], p["random_higher"]) == (2, 1, 1)


# ---- selection and the invariants -------------------------------------------------------------------

def _man_row(acc, arch, dt, payload_aa, orf_aa, start, end):
    return dict(filename=f"{acc}.gbk", construct_id=f"syn_{acc}_{arch}_{int(float(dt) * 100):02d}",
                archetype_manual=arch, payload_accession=acc, payload_aa=str(payload_aa),
                plasmid_bp="5000", orf_start_nt="1", orf_aa=str(orf_aa),
                payload_start_aa=str(start), payload_end_aa=str(end),
                dilution_target=dt, dilution_actual=str(round(1 - payload_aa / orf_aa, 4)),
                payload_fraction="0.3588", parts="payload", label_source="exact_by_construction",
                annotation_audited="n/a")


def _manifest(accs, lengths=None):
    lengths = lengths or {}
    out = {}
    for acc in accs:
        p = lengths.get(acc, 100)
        for arch in ("motifless_fusion", "tagged_fusion"):
            for dt, orf in (("0.6", p * 2), ("0.8", p * 5)):
                if arch == "motifless_fusion":
                    start, end = 1, p
                else:
                    start, end = orf - p + 1, orf
                r = _man_row(acc, arch, dt, p, orf, start, end)
                out[r["construct_id"]] = r
    return out


def test_selection_is_paired_deterministic_and_a_prefix():
    man = _manifest(["P3", "P1", "P2"])
    sel = ct.select_constructs(man, "motifless_fusion", ["0.8", "0.6"])
    assert len(sel) == 6
    assert [c["payload_accession"] for c in sel] == ["P1", "P1", "P2", "P2", "P3", "P3"]
    assert [c["level"] for c in sel[:2]] == ["0.8", "0.6"]
    small = ct.select_constructs(man, "motifless_fusion", ["0.8", "0.6"], max_payloads=2)
    assert [c["construct_id"] for c in small] == [c["construct_id"] for c in sel[:4]]


def test_a_payload_missing_a_level_is_dropped_from_both_levels():
    man = _manifest(["P1", "P2"])
    del man["syn_P2_motifless_fusion_80"]
    sel = ct.select_constructs(man, "motifless_fusion", ["0.8", "0.6"])
    assert {c["payload_accession"] for c in sel} == {"P1"}


def test_selection_recomputes_f_rather_than_reading_the_column():
    man = _manifest(["P1"], lengths={"P1": 200})
    sel = ct.select_constructs(man, "motifless_fusion", ["0.8"])
    assert sel[0]["f"] == pytest.approx(0.2)


def test_manifest_mismatch_is_detected_field_by_field():
    a = _manifest(["P1", "P2"])
    b = _manifest(["P1", "P2"])
    assert ct.check_manifests_match(a, b) == []
    b["syn_P2_motifless_fusion_80"]["orf_aa"] = "999"
    bad = ct.check_manifests_match(a, b)
    assert len(bad) == 1 and "orf_aa" in bad[0]
    del b["syn_P1_motifless_fusion_60"]
    assert any("construct id sets differ" in x for x in ct.check_manifests_match(a, b))


# ---- end to end, with a deterministic stand-in for a 2.6 GB download --------------------------------

class _StubEmbedder:
    """Per-residue vectors that are a function of the residue AND of a declared context radius.

    `corrupt` is how strongly the rest of the sequence perturbs a residue's own vector, so one stub drives
    both hypotheses: `corrupt=0.0` is pure displacement (a residue's vector depends only on itself, so
    the payload's rows are bit-identical in every context and only the pooled mean moves), and a large
    `corrupt` is cross-talk. This is what makes the end-to-end test a test of the *verdict* and not just
    of the CSV writer.

    `radius=None` makes the context **global**, which is the geometry ESM-2 actually has: attention is
    over the whole sequence, so a filler residue 2,000 positions away can reach the payload. A finite
    radius confines the effect to the junction, and the two are different findings, which is why
    `JUNCTION_AA` exists in the script and why both are exercised here.

    It truncates at `max_length` like the real embedder, so the script's truncation path is reachable.
    """

    name = "stub"

    def __init__(self, dim: int = 24, corrupt: float = 0.0, radius: int | None = None,
                 max_length: int = 2048) -> None:
        self.dim, self.corrupt, self.radius = dim, corrupt, radius
        self.max_length = max_length
        self.n_truncated = 0
        rng = np.random.default_rng(6103)
        self._table = {a: rng.normal(size=dim) for a in AA}
        self._zero = np.zeros(dim)

    def _rows(self, seq: str) -> np.ndarray:
        base = np.stack([self._table.get(c, self._zero) for c in seq])
        if self.corrupt:
            if self.radius is None:
                ctx = np.broadcast_to(base.mean(axis=0), base.shape).copy()
            else:
                pad = np.zeros((self.radius, self.dim))
                padded = np.concatenate([pad, base, pad])
                ctx = np.stack([padded[i: i + 2 * self.radius + 1].mean(axis=0)
                                for i in range(len(seq))])
            # The context is applied as a *unit direction* times a multiple of the typical residue norm,
            # so `corrupt` is in units of "fraction of a residue vector" and the test can ask for a
            # perturbation of a stated size. Taking the raw mean instead makes the effect vanish for a
            # long random sequence, whose mean is near zero, which is a property of the stub and not of
            # any hypothesis, and it is how this fixture first produced a misleading "neither".
            n = np.linalg.norm(ctx, axis=1, keepdims=True)
            unit = ctx / np.maximum(n, 1e-12)
            scale = self.corrupt * float(np.linalg.norm(base, axis=1).mean())
            base = base + scale * unit
        return base.astype(np.float32)

    def embed_residues_pooled(self, seqs):
        l2, raw, pooled = [], [], []
        for s in seqs:
            if len(s) > self.max_length:
                self.n_truncated += 1
            s = s[: self.max_length]
            r = self._rows(s)
            raw.append(r)
            n = np.linalg.norm(r, axis=1, keepdims=True)
            l2.append((r / np.maximum(n, 1e-12)).astype(np.float32))
            m = r.mean(axis=0)
            pooled.append(m / max(float(np.linalg.norm(m)), 1e-12))
        return l2, raw, np.stack(pooled).astype(np.float32)


@pytest.fixture
def tier(tmp_path):
    """Two tiers that are the same constructs with different filler, written out as the real ones are."""
    rng = np.random.default_rng(610)

    def seq(n):
        return "".join(AA[i] for i in rng.integers(0, 20, size=n))

    accs = ["P000A1", "P000B2", "P000C3"]
    payloads = {a: seq(60) for a in accs}
    man_rows, orf_text = [], {"random": [], "real": []}
    for acc in accs:
        p = payloads[acc]
        for dt, mult in (("0.6", 2), ("0.8", 5)):
            orf_len = len(p) * mult
            cid = f"syn_{acc}_motifless_fusion_{int(float(dt) * 100):02d}"
            man_rows.append(_man_row(acc, "motifless_fusion", dt, len(p), orf_len, 1, len(p)))
            for filler in ("random", "real"):
                # The filler differs; the payload prefix does not. That is 60.1's invariant.
                tail = seq(orf_len - len(p)) if filler == "random" else ("GAVLIP" * orf_len)[
                    : orf_len - len(p)]
                orf_text[filler].append(f">{cid}|orf001|+|1..9 length_aa={orf_len}\n{p + tail}\n")
                # a decoy ORF, so pick_payload_orf has to choose
                orf_text[filler].append(f">{cid}|orf002|+|1..9 length_aa=80\n{seq(80)}\n")
    paths = {}
    for filler in ("random", "real"):
        d = tmp_path / filler
        d.mkdir()
        (d / "orfs.faa").write_text("".join(orf_text[filler]), encoding="utf-8")
        mp = d / "synthetic_manifest.csv"
        with open(mp, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(man_rows[0]))
            w.writeheader()
            w.writerows(man_rows)
        paths[filler] = (mp, d / "orfs.faa")
    ref = tmp_path / "ref.fasta"
    ref.write_text("".join(f">sp|{a}|X\n{payloads[a]}\n" for a in accs), encoding="utf-8")
    return paths, ref, accs


def _argv(paths, ref, out_dir, levels=("0.8", "0.6")):
    return ["--manifest-random", str(paths["random"][0]), "--orfs-random", str(paths["random"][1]),
            "--manifest-real", str(paths["real"][0]), "--orfs-real", str(paths["real"][1]),
            "--reference", str(ref), "--out-dir", str(out_dir), "--embedder", "stub-esm2",
            "--levels", *levels, "--allow-md5-mismatch"]


def _run(monkeypatch, paths, ref, out_dir, corrupt, levels=("0.8", "0.6"), radius=None, extra=()):
    monkeypatch.setattr(
        ct, "make_embedder",
        lambda spec, batch_size, max_length: _StubEmbedder(corrupt=corrupt, radius=radius,
                                                           max_length=max_length))
    rc = ct.main(_argv(paths, ref, out_dir, levels) + list(extra))
    assert rc == 0
    blob = json.loads((out_dir / "crosstalk_stub-esm2_summary.json").read_text(encoding="utf-8"))
    with open(out_dir / "crosstalk_stub-esm2.csv", newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    return blob, rows


def _at(blob, level):
    return next(v for v in blob["verdicts"] if v["level"] == level)


def test_end_to_end_a_context_free_embedder_reads_as_displacement(monkeypatch, tier, tmp_path):
    """With `corrupt=0` the payload's rows cannot depend on the filler, so the only honest answer is
    displacement, and the pooled cosine still falls a long way, which is the point."""
    paths, ref, accs = tier
    blob, rows = _run(monkeypatch, paths, ref, tmp_path / "out", corrupt=0.0)
    # The label is asserted at the most diluted level, where the pooled vector has moved enough for
    # there to be a mechanism to name. The milder level reading `neither` is the rule working: a stub
    # whose pooled movement is below the magnitude gate must not be handed a mechanism.
    assert _at(blob, "0.8")["label"] == "displacement"
    assert _at(blob, "0.8")["crosstalk_share_of_deficit"] == pytest.approx(0.0, abs=1e-4)
    for s in blob["by_level_and_filler"]:
        assert s["cos_res_mean"] == pytest.approx(1.0, abs=1e-5)
        assert s["pooled_cos_whole_res"] < 0.99
    assert all(r["measurable"] == "True" for r in rows)
    assert len(rows) == 2 * len(accs) * 2


def test_end_to_end_a_globally_context_sensitive_embedder_reads_as_crosstalk_or_both(
        monkeypatch, tier, tmp_path):
    """Global context, the geometry ESM-2 has: the filler reaches every payload residue."""
    paths, ref, accs = tier
    blob, _ = _run(monkeypatch, paths, ref, tmp_path / "out2", corrupt=1.5)
    assert _at(blob, "0.8")["label"] in {"crosstalk", "both"}, _at(blob, "0.8")
    rnd = [s for s in blob["by_level_and_filler"] if s["filler"] == "random"]
    assert all(s["cos_res_mean"] < 0.99 for s in rnd)


def test_junction_and_core_separate_a_boundary_effect_from_a_payload_wide_one(
        monkeypatch, tier, tmp_path):
    """A finite context radius must show up as a junction effect with an intact core, which is a
    different finding from cross-talk and must not be reported as the same thing."""
    paths, ref, accs = tier
    blob, _ = _run(monkeypatch, paths, ref, tmp_path / "out7", corrupt=2.0, radius=4,
                   levels=("0.8",))
    for s in blob["by_level_and_filler"]:
        assert s["cos_res_mean_junction"] < s["cos_res_mean_core"]
        assert s["cos_res_mean_core"] == pytest.approx(1.0, abs=1e-4)


def test_end_to_end_emits_the_per_residue_distribution_not_just_its_mean(monkeypatch, tier, tmp_path):
    paths, ref, accs = tier
    out = tmp_path / "out3"
    _run(monkeypatch, paths, ref, out, corrupt=0.5, levels=("0.8",))
    with open(out / "crosstalk_stub-esm2_residues.csv", newline="", encoding="utf-8-sig") as fh:
        res = list(csv.DictReader(fh))
    # 3 payloads x 1 level x 2 fillers x 60 residues
    assert len(res) == 3 * 2 * 60
    assert {r["filler"] for r in res} == {"random", "real"}
    assert all(-1.0 <= float(r["cosine"]) <= 1.0 for r in res)


def test_a_wrong_orfs_faa_refuses_to_run(monkeypatch, tier, tmp_path):
    """the md5 is the only thing that establishes which tier a file is."""
    paths, ref, _ = tier
    monkeypatch.setattr(ct, "make_embedder", lambda *a, **k: _StubEmbedder())
    argv = [a for a in _argv(paths, ref, tmp_path / "out4") if a != "--allow-md5-mismatch"]
    assert ct.main(argv) == 2


def test_a_manifest_disagreement_stops_the_run(monkeypatch, tier, tmp_path):
    paths, ref, _ = tier
    rows = list(csv.DictReader(open(paths["real"][0], newline="", encoding="utf-8-sig")))
    rows[0]["orf_aa"] = "7"
    with open(paths["real"][0], "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    monkeypatch.setattr(ct, "make_embedder", lambda *a, **k: _StubEmbedder())
    assert ct.main(_argv(paths, ref, tmp_path / "out5")) == 2


def test_a_payload_past_max_length_is_reported_unmeasurable(monkeypatch, tier, tmp_path):
    """Truncation is counted, never slice-fitted. The row survives with a reason so the denominator is
    visible instead of quietly shrinking."""
    paths, ref, _ = tier
    out = tmp_path / "out6"
    # The payload is 60 aa at the ORF's N-terminus, so a 30 aa cap keeps half of it: the oracle query is
    # cut to 30 rows and the in-ORF payload span to 30 as well, and the script must refuse the comparison
    # rather than quietly comparing 30 residues and calling it the payload.
    blob, rows = _run(monkeypatch, paths, ref, out, corrupt=0.0, levels=("0.8",),
                      extra=("--max-length", "30"))
    assert rows and all(r["measurable"] == "False" for r in rows)
    assert all("max-length" in r["skip_reason"] or "do not match" in r["skip_reason"] for r in rows)
    assert blob["n_measurable"] == 0
