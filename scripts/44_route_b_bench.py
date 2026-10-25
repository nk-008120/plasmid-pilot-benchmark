"""Turn the route B f measurement into a benchmark tier the retrieval runner can score.

Script 43 measured f for every (deposit, payload) pair. This script makes the three things
`10_twilight.py` needs and nothing else:

  data/route_b_kept/        one GenBank record per deposit: the record 43's dedupe chose, so the
                            retrieval tier and the f table describe the same molecules. Copying
                            rather than globbing the raw directory is deliberate: `data/route_b_raw`
                            holds five superseded records, and a retrieval run that silently picked
                            a different one than the f table did would be the coverage-denominator
                            bug in yet another costume.
  data/manifest_route_b.csv        filename -> payload_accession, the primary payload per file.
  data/manifest_route_b_second.csv the four files that carry a SECOND selected payload.

The manifest is keyed by filename and the runner reads exactly one `payload_accession` per file, so
a construct carrying two selected payloads cannot be scored for both in one pass. Four deposits are
in that position (176139/176140/176141 carry P06746 and P18887; 193356 carries P48431 and Q01860).
Dropping the second payload would quietly shrink the tier, and choosing which of the two to keep by
its f would select on the quantity under study, so the second payloads go in their own manifest
and get their own pass, and the two are merged at reporting time. The split is alphabetical by
accession, which is a property of the name and not of the result.

ABSENT and ENGINEERED VARIANT pairs are excluded: an absent payload has no ground truth to retrieve,
and whether an engineered variant counts as its payload is a judgement this project's working rules reserves to the user.
Both are written to the manifest with a `class` column so the exclusion is visible rather than
implied, and `--include-variants` scores them if the user decides they belong.
"""
import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_COLS = ["filename", "source_id", "payload_accession", "payload_description",
                 "f", "stratum", "group", "class"]


def read_f_table(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def partition(rows: list[dict], include_variants: bool = False) -> tuple[list[dict], list[dict], list[dict]]:
    """Split the f table into (primary, second, excluded) manifest rows.

    A file's payloads are ordered by accession, so which one is 'primary' is fixed by the name and
    cannot drift with the measurement.
    """
    keep_classes = {"canonical"} | ({"engineered_variant"} if include_variants else set())
    by_file: dict[str, list[dict]] = defaultdict(list)
    excluded = []
    for r in rows:
        if not r.get("f"):
            excluded.append(r)
        elif r.get("payload_present") in keep_classes:
            by_file[r["file"]].append(r)
        else:
            excluded.append(r)
    primary, second = [], []
    for fn in sorted(by_file):
        for i, r in enumerate(sorted(by_file[fn], key=lambda x: x["payload_accession"])):
            (primary if i == 0 else second).append(r)
    return primary, second, excluded


def manifest_row(r: dict) -> dict:
    return {"filename": r["file"], "source_id": f"addgene_{r['addgene_id']}",
            "payload_accession": r["payload_accession"],
            "payload_description": r["payload_name"], "f": r["f"],
            "stratum": r["stratum"], "group": r["group"], "class": r["payload_present"]}


def write_manifest(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLS)
        w.writeheader()
        w.writerows(manifest_row(r) for r in rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--f-table", type=Path, default=ROOT / "results/route_b_f/f_by_construct.csv")
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data/route_b_raw")
    ap.add_argument("--kept-dir", type=Path, default=ROOT / "data/route_b_kept")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/manifest_route_b.csv")
    ap.add_argument("--manifest-second", type=Path,
                    default=ROOT / "data/manifest_route_b_second.csv")
    ap.add_argument("--include-variants", action="store_true",
                    help="score the four engineered-variant pairs as their payload (the user's call)")
    args = ap.parse_args(argv)

    rows = read_f_table(args.f_table)
    primary, second, excluded = partition(rows, args.include_variants)

    args.kept_dir.mkdir(parents=True, exist_ok=True)
    for old in args.kept_dir.glob("*.gbk"):
        old.unlink()
    kept_files = sorted({r["file"] for r in primary} | {r["file"] for r in second})
    for fn in kept_files:
        shutil.copyfile(args.raw_dir / fn, args.kept_dir / fn)

    write_manifest(args.manifest, primary)
    write_manifest(args.manifest_second, second)

    print(f"route B benchmark tier")
    print(f"  kept records: {len(kept_files)} copied to {args.kept_dir}")
    print(f"  BINDING, one payload per file: {len(second)} second payload(s) moved to a second pass "
          f"{sorted({(r['addgene_id'], r['payload_accession']) for r in second})}")
    print(f"  BINDING, excluded as not canonical: {len(excluded)} pair(s) "
          f"{sorted({(r['addgene_id'], r['payload_accession'], r['payload_present']) for r in excluded})}")
    print(f"  primary manifest: {len(primary)} rows -> {args.manifest}")
    strata: dict[str, int] = defaultdict(int)
    for r in primary + second:
        strata[r["stratum"]] += 1
    print(f"  f strata over {len(primary) + len(second)} scored pairs: "
          + "  ".join(f"{k}  {strata[k]}" for k in ("<0.50", "0.50-0.70", "0.70-0.95", ">=0.95")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
