"""Step 1 runner: load plasmids, call ORFs, score them against annotated CDSs, write tables + provenance."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import Bio
from Bio.Seq import Seq

from . import __version__
from .archetype import classify_archetype
from .genbank import GENBANK_SUFFIXES, annotated_cds, load_plasmids, sha256_file
from .insert import check_insert, parse_insert_size
from .intervals import crosses_origin
from .orfs import find_orfs
from .recovery import analyse

CDS_STATUSES = ("exact", "contained", "truncated", "missed", "below_min_aa", "invalid")


def _fmt_segments(segs: list[tuple[int, int]]) -> str:
    return ";".join(f"{s + 1}..{e}" for s, e in segs)


def _orf_id(plasmid_id: str, idx: int, orf, L: int) -> str:
    sign = "+" if orf.strand > 0 else "-"
    return f"{plasmid_id}|orf{idx + 1:03d}|{sign}|{orf.start + 1}..{orf.end(L)}"


def _read_manifest(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    # utf-8-sig: Excel's "CSV UTF-8" adds a BOM that would otherwise corrupt the first column name and
    # make every row silently disappear. errors="replace" keeps an Excel "CSV (Comma delimited)" file readable.
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.DictReader(handle)
        reader.fieldnames = [f.strip() for f in (reader.fieldnames or [])]
        if "filename" not in reader.fieldnames:
            raise SystemExit(f"manifest {path} has no 'filename' column (found: {reader.fieldnames}); check the header row")
        return {row["filename"].strip(): row for row in reader if (row.get("filename") or "").strip()}


def _git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
        return out.stdout.strip()
    except Exception:
        return None


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    # utf-8-sig so Excel on Windows shows non-ASCII labels (e.g. lacZ-alpha) correctly; csv readers should use utf-8-sig.
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(
    input_dir: Path,
    out_dir: Path,
    min_aa: int = 50,
    start_codons: tuple[str, ...] = ("ATG",),
    assume_topology: str = "circular",
    manifest: Path | None = None,
    argv: list[str] | None = None,
    backbone_threshold: float = 0.15,
    insert_tolerance: float = 0.10,
) -> dict:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in GENBANK_SUFFIXES)
    manifest_rows = _read_manifest(manifest)
    file_names = {p.name for p in files}
    manifest_without_file = sorted(set(manifest_rows) - file_names) if manifest is not None else []
    files_without_manifest = sorted(file_names - set(manifest_rows)) if manifest is not None else []

    summary_rows, cds_rows, orf_rows, fasta, errors, inputs = [], [], [], [], [], []
    seen_seq: dict[str, str] = {}
    for path in files:
        inputs.append({"file": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
        try:
            plasmids = load_plasmids(path, assume_topology=assume_topology)
        except Exception as exc:
            errors.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if not plasmids:
            errors.append({"file": path.name, "error": "no GenBank records found"})
            continue

        for pl in plasmids:
            L = pl.length
            cdss = annotated_cds(pl)
            orfs = find_orfs(pl.seq, circular=pl.circular, min_aa=min_aa, start_codons=start_codons)
            cds_res, orf_res = analyse(pl, cdss, orfs, min_aa)
            arch = classify_archetype(pl, cdss, min_aa=min_aa, backbone_threshold=backbone_threshold)

            man = manifest_rows.get(pl.source_file, {})
            ins = check_insert(orfs, parse_insert_size(man.get("insert_size_bp")), insert_tolerance)
            canon = min(pl.seq, str(Seq(pl.seq).reverse_complement()))
            digest = hashlib.md5(canon.encode()).hexdigest()
            duplicate_of = seen_seq.setdefault(digest, pl.id)
            duplicate_of = "" if duplicate_of == pl.id else duplicate_of

            status = Counter(r.status for r in cds_res)
            klass = Counter(r.klass for r in orf_res)
            warnings = list(pl.warnings)
            if pl.topology_assumed:
                warnings.append(f"topology missing in file; assumed {pl.topology}")
            if pl.n_nonacgt:
                warnings.append(f"{pl.n_nonacgt} non-ACGT bases")

            summary_rows.append(
                {
                    "plasmid_id": pl.id,
                    "source_file": pl.source_file,
                    "name": pl.name,
                    "length_bp": L,
                    "topology": pl.topology,
                    "topology_assumed": pl.topology_assumed,
                    "gc_fraction": round(pl.gc_fraction, 4),
                    "n_nonacgt": pl.n_nonacgt,
                    "duplicate_of": duplicate_of,
                    "archetype_auto": arch.archetype,
                    "archetype_manual": man.get("archetype_manual", ""),
                    "n_payload_cds": arch.n_payload_cds,
                    "payload_fraction": arch.payload_fraction,
                    "tag_near_cds": arch.tag_near_cds,
                    "tag_labels": ";".join(arch.tag_labels),
                    "backbone_heavy": arch.backbone_heavy,
                    "source_id": man.get("source_id", ""),
                    "source_url": man.get("source_url", ""),
                    "annotation_audited": man.get("annotation_audited", ""),
                    "manifest_notes": man.get("notes", ""),
                    "n_features": len(pl.features),
                    "n_cds_annotated": len(cdss),
                    **{f"cds_{s}": status.get(s, 0) for s in CDS_STATUSES},
                    "orfs_total": len(orfs),
                    "orfs_cds_match": klass.get("cds_match", 0),
                    "orfs_cds_shadow": klass.get("cds_shadow", 0),
                    "orfs_unannotated": klass.get("unannotated", 0),
                    "orf_precision": round(klass.get("cds_match", 0) / len(orfs), 4) if orfs else "",
                    "insert_size_bp": ins.insert_size_bp if ins.insert_size_bp is not None else "",
                    "insert_best_orf_nt": ins.best_orf_nt if ins.best_orf_nt is not None else "",
                    "insert_ratio": ins.ratio if ins.ratio is not None else "",
                    "insert_verdict": ins.verdict,
                    "insert_best_orf_class": orf_res[ins.best_orf_index].klass if ins.best_orf_index is not None else "",
                    "insert_best_orf_id": _orf_id(pl.id, ins.best_orf_index, orfs[ins.best_orf_index], L) if ins.best_orf_index is not None else "",
                    "warnings": " | ".join(warnings),
                }
            )

            for r in cds_res:
                c = r.cds
                cds_rows.append(
                    {
                        "plasmid_id": pl.id,
                        "cds_label": c.label,
                        "strand": c.strand,
                        "location_1based": _fmt_segments(c.segments),
                        "wraps_origin": crosses_origin(c.segments, L),
                        "n_segments": len(c.segments),
                        "length_aa": len(c.protein),
                        "has_terminal_stop": c.has_terminal_stop,
                        "status": r.status,
                        "invalid_reason": c.invalid_reason,
                        "orf_id": _orf_id(pl.id, r.orf_index, orfs[r.orf_index], L) if r.orf_index is not None else "",
                        "orf_minus_cds_aa": "" if r.orf_minus_cds_aa is None else r.orf_minus_cds_aa,
                        "n_extra_aa": "" if r.n_extra_aa is None else r.n_extra_aa,
                        "c_extra_aa": "" if r.c_extra_aa is None else r.c_extra_aa,
                        "coverage": "" if r.coverage is None else r.coverage,
                    }
                )

            for r in orf_res:
                o = r.orf
                oid = _orf_id(pl.id, r.index, o, L)
                orf_rows.append(
                    {
                        "orf_id": oid,
                        "plasmid_id": pl.id,
                        "strand": o.strand,
                        "start_1based": o.start + 1,
                        "end_1based": o.end(L),
                        "length_aa": o.length_aa,
                        "wraps_origin": o.wraps,
                        "class": r.klass,
                        "matched_cds": ";".join(r.matched_cds),
                        "feature_context": r.feature_context,
                    }
                )
                fasta.append(f">{oid} length_aa={o.length_aa} class={r.klass}\n{o.protein}\n")

    summary_fields = list(summary_rows[0].keys()) if summary_rows else []
    cds_fields = [
        "plasmid_id", "cds_label", "strand", "location_1based", "wraps_origin", "n_segments", "length_aa",
        "has_terminal_stop", "status", "invalid_reason", "orf_id", "orf_minus_cds_aa",
        "n_extra_aa", "c_extra_aa", "coverage",
    ]
    orf_fields = [
        "orf_id", "plasmid_id", "strand", "start_1based", "end_1based", "length_aa",
        "wraps_origin", "class", "matched_cds", "feature_context",
    ]
    _write_csv(out_dir / "pilot_summary.csv", summary_rows, summary_fields)
    _write_csv(out_dir / "cds_recovery.csv", cds_rows, cds_fields)
    _write_csv(out_dir / "orf_calls.csv", orf_rows, orf_fields)
    (out_dir / "orfs.faa").write_text("".join(fasta), encoding="utf-8")

    cds_counts = Counter(r["status"] for r in cds_rows)
    orf_counts = Counter(r["class"] for r in orf_rows)
    scorable = sum(cds_counts.get(s, 0) for s in ("exact", "contained", "truncated", "missed"))
    aggregate = {
        "n_files": len(files),
        "n_plasmids": len(summary_rows),
        "n_load_errors": len(errors),
        "cds_status_counts": dict(cds_counts),
        "scorable_cds": scorable,
        "orf_class_counts": dict(orf_counts),
        "archetype_auto_counts": dict(Counter(r["archetype_auto"] for r in summary_rows)),
        "insert_verdict_counts": dict(Counter(r["insert_verdict"] for r in summary_rows if r["insert_verdict"])),
        "duplicates": [r["plasmid_id"] for r in summary_rows if r["duplicate_of"]],
        "manifest_rows_without_file": manifest_without_file,
        "files_without_manifest_row": files_without_manifest,
    }
    run_info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_seconds": round(time.time() - t0, 2),
        "package_version": __version__,
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "python": sys.version.split()[0],
        "biopython": Bio.__version__,
        "platform": platform.platform(),
        "argv": argv if argv is not None else sys.argv[1:],
        "parameters": {
            "min_aa": min_aa,
            "start_codons": list(start_codons),
            "assume_topology": assume_topology,
            "backbone_threshold": backbone_threshold,
            "insert_tolerance": insert_tolerance,
            "manifest": str(manifest) if manifest else None,
        },
        "inputs": inputs,
        "load_errors": errors,
        "aggregate": aggregate,
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    return {"summary_rows": summary_rows, "aggregate": aggregate, "errors": errors}


def _print_report(result: dict, min_aa: int, start_codons: tuple[str, ...]) -> None:
    rows, agg = result["summary_rows"], result["aggregate"]
    print(f"\nORF calling: min {min_aa} aa, start codons {','.join(start_codons)}")
    print(f"Plasmids loaded: {agg['n_plasmids']} (load errors: {agg['n_load_errors']})")
    for err in result["errors"]:
        print(f"  ! {err['file']}: {err['error']}")
    if not rows:
        return
    header = f"{'plasmid':<28}{'bp':>7}  {'archetype':<20}{'cds':>4}{'exact':>6}{'cont':>5}{'trunc':>6}{'miss':>5}{'orfs':>6}{'prec':>7}"
    print("\n" + header + "\n" + "-" * len(header))
    for r in rows:
        prec = f"{r['orf_precision']:.2f}" if r["orf_precision"] != "" else "  -"
        arch = (r["archetype_manual"] or r["archetype_auto"])[:19]
        print(
            f"{r['plasmid_id'][:27]:<28}{r['length_bp']:>7}  {arch:<20}{r['n_cds_annotated']:>4}"
            f"{r['cds_exact']:>6}{r['cds_contained']:>5}{r['cds_truncated']:>6}{r['cds_missed']:>5}"
            f"{r['orfs_total']:>6}{prec:>7}"
        )
    c, n = agg["cds_status_counts"], agg["scorable_cds"]
    if n:
        print(
            f"\nScorable annotated CDSs: {n}  |  exact {c.get('exact', 0)}  contained {c.get('contained', 0)}"
            f"  truncated {c.get('truncated', 0)}  missed {c.get('missed', 0)}"
        )
    print(f"Excluded: below_min_aa {c.get('below_min_aa', 0)}, invalid annotation {c.get('invalid', 0)}")
    o = agg["orf_class_counts"]
    print(
        f"Called ORFs: {sum(o.values())}  |  cds_match {o.get('cds_match', 0)}    "
        f"cds_shadow {o.get('cds_shadow', 0)}  unannotated {o.get('unannotated', 0)}"
    )
    iv = agg["insert_verdict_counts"]
    if iv:
        stated = sum(iv.values())
        print(
            f"Insert-size check ({stated} plasmids with a stated insert): recovered {iv.get('recovered', 0)}"
            f"  short {iv.get('short', 0)}  long {iv.get('long', 0)}  no_orfs {iv.get('no_orfs', 0)}"
        )
    if agg["duplicates"]:
        print(f"Duplicate sequences flagged: {', '.join(agg['duplicates'])}")
    if agg["manifest_rows_without_file"]:
        print(f"WARNING manifest rows with no matching file (check spelling): {', '.join(agg['manifest_rows_without_file'])}")
    if agg["files_without_manifest_row"]:
        print(f"WARNING files with no manifest row: {', '.join(agg['files_without_manifest_row'])}")
    print("Archetypes (auto):", ", ".join(f"{k}={v}" for k, v in sorted(agg["archetype_auto_counts"].items())))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, default=Path("data/raw"), help="folder of GenBank files")
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--min-aa", type=int, default=50, help="minimum ORF length in amino acids (default 50)")
    ap.add_argument("--start-codons", default="ATG", help="comma-separated, e.g. ATG,GTG,TTG (default ATG)")
    ap.add_argument("--assume-topology", choices=("circular", "linear"), default="circular")
    ap.add_argument("--manifest", type=Path, default=None, help="optional CSV (see data/manifest_template.csv)")
    ap.add_argument(
        "--insert-tolerance", type=float, default=0.10,
        help="fraction by which a called ORF may differ from the manifest's insert_size_bp and still count as recovered",
    )
    ap.add_argument(
        "--backbone-threshold", type=float, default=0.15,
        help="a single-payload plasmid whose payload CDS covers less than this fraction is labelled backbone_heavy",
    )
    args = ap.parse_args(argv)

    if not args.input_dir.is_dir():
        print(f"input directory not found: {args.input_dir}", file=sys.stderr)
        return 2
    starts = tuple(s.strip().upper() for s in args.start_codons.split(",") if s.strip())
    result = run(
        args.input_dir, args.out_dir, args.min_aa, starts, args.assume_topology, args.manifest,
        argv if argv is not None else sys.argv[1:], args.backbone_threshold, args.insert_tolerance,
    )
    _print_report(result, args.min_aa, starts)
    print(f"\nWrote tables, orfs.faa and run_info.json to {args.out_dir}/")
    return 0 if result["summary_rows"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
