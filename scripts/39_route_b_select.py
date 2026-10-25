"""Write the Route B payload selection, with every number pulled from the measurements.

Nothing here is transcribed by hand: the gate counts come from
`results/route_b_homologs/payload_lengths.csv` and the deposit/architecture fields from
`data/route_b_candidates.csv`, so a typo in the pick list cannot survive this script.
"""
import csv
from pathlib import Path

PICKS = [
    # accession, group, why this one
    ("O43474", "A reprogramming cassette", "OSKM member; 12 deposits, 2A and IRES and polycistronic"),
    ("P01106", "A reprogramming cassette", "OSKM member; polycistronic deposits present"),
    ("P48431", "A reprogramming cassette", "OSKM member; the highest deposit count of the four"),
    ("Q01860", "A reprogramming cassette", "OSKM member; completes the cassette. THIN at 25% - read at 35%"),
    ("Q13562", "A reprogramming cassette", "neuronal reprogramming cassettes"),
    ("P61371", "A reprogramming cassette", "motor-neuron programming cassettes, usually with LHX3"),
    ("Q08369", "A reprogramming cassette", "cardiac GATA4/MEF2C/TBX5 cassette"),
    ("P04386", "B standard tool", "47 deposits, the most in the harvest; 2A and IRES both"),
    ("J7RUA5", "B standard tool", "compact Cas9 for AAV; distinct family from SpCas9. TRUNCATED at 1022"),
    ("P00573", "B standard tool", "classic enzyme, non-mammalian; thin family at 25%"),
    ("P06746", "B standard tool", "DNA polymerase beta, standard enzyme"),
    ("P18887", "B standard tool", "DNA repair, moderate family, 2A deposits"),
    ("Q96524", "B standard tool", "optogenetic tool; plant, so it widens species coverage"),
    ("P20309", "C family representative", "GPCR representative: lowest 25% family of its 9-member cluster"),
    ("P36888", "C family representative", "kinase representative: lowest 25% family of its 17-member cluster"),
    ("P06536", "C family representative", "nuclear receptor representative; IRES-heavy deposits"),
    ("P01116", "C family representative", "small GTPase representative; 189 aa, so a low f is easy to reach"),
    ("P36544", "C family representative", "ligand-gated ion channel representative; polycistronic deposits"),
    ("P97438", "D channel / transporter", "10 deposits, IRES in all of them"),
    ("Q9H2V7", "D channel / transporter", "transporter, IRES deposits, healthy remote family"),
    ("P60709", "D channel / transporter", "housekeeping; the easy-retrieval control of the set"),
]

root = Path(".")
cand = {}
for r in csv.DictReader(open(root / "data/route_b_candidates.csv", encoding="utf-8-sig")):
    a = (r["uniprot_accession"] or "").strip()
    if a:
        cand[a] = r
gate = {r["payload_accession"]: r
        for r in csv.DictReader(open(root / "results/route_b_homologs/payload_lengths.csv",
                                      encoding="utf-8-sig"))}

out = root / "data/route_b_selected.csv"
cols = ["uniprot_accession", "protein_name", "species", "length_aa", "group",
        "n_deposits", "architectures", "n_homologs", "available_at_35", "available_at_25",
        "esm2_truncated_at_2048", "why_selected"]
missing = [a for a, *_ in PICKS if a not in cand or a not in gate]
if missing:
    raise SystemExit(f"ABORT: not in the menu or not searched: {missing}")

with open(out, "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for acc, group, why in PICKS:
        c, g = cand[acc], gate[acc]
        w.writerow({
            "uniprot_accession": acc,
            "protein_name": c["protein_name"],
            "species": c["species"],
            "length_aa": g["payload_aa"],
            "group": group,
            "n_deposits": c["n_deposits"],
            "architectures": c["architectures"],
            "n_homologs": g["n_homologs"],
            "available_at_35": g["available_at_35"],
            "available_at_25": g["available_at_25"],
            "esm2_truncated_at_2048": int(g["payload_aa"]) > 2048,
            "why_selected": why,
        })

rows = list(csv.DictReader(open(out, encoding="utf-8-sig")))
print(f"wrote {out}  ({len(rows)} payloads)\n")
hdr = f"{'acc':<8}{'aa':>6}{'dep':>5}{'n35':>6}{'n25':>6}    {'group':<26}{'name':<44}"
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r['uniprot_accession']:<8}{r['length_aa']:>6}{r['n_deposits']:>5}"
          f"{r['available_at_35']:>6}{r['available_at_25']:>6}    {r['group']:<26}{r['protein_name'][:44]:<44}")

n25 = [int(r["available_at_25"]) for r in rows]
print(f"\ndeposits represented: {sum(int(r['n_deposits']) for r in rows)}")
print(f"25% gate: all {len(rows)} scorable; weakest {min(n25)}, median {sorted(n25)[len(n25)//2]}, strongest {max(n25)}")
print(f"under 4 remote homologs at 25% (read these at 35%): "
      f"{[r['uniprot_accession'] for r in rows if int(r['available_at_25']) < 4]}")
print(f"over 1022 aa (ESM-2 truncation applies): "
      f"{[(r['uniprot_accession'], r['length_aa']) for r in rows if int(r['length_aa']) > 1022]}")
arch = {"2A": 0, "IRES": 0, "polycistronic": 0}
for r in rows:
    for k in arch:
        if k in r["architectures"]:
            arch[k] += 1
print(f"architecture coverage (deposit-level flags, not protein-level): {arch}")
