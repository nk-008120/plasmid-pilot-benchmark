#!/usr/bin/env bash
# Phase 2 of the twilight reference build: align every tier's payloads against Swiss-Prot.
#
#   bash /mnt/c/NK/plasmid-pilot/scripts/payload_homologs.sh
#
# Finds DIAMOND wherever it ended up, so it works in a shell that predates the PATH change. Idempotent:


REPO=/mnt/c/NK/plasmid-pilot
LOG="$REPO/results/payload_homologs.log"
Q="$REPO/data/reference/payloads.fasta"
OUT="$REPO/data/reference/payload_hits.tsv"
mkdir -p "$REPO/results"
exec > >(tee "$LOG") 2>&1
echo "=== payload_homologs.sh  $(date -Is) ==="

# DIAMOND may be on PATH, in ~/bin from the earlier setup, or installed via apt as diamond-aligner.
DMND=""
for cand in diamond "$HOME/bin/diamond" /usr/local/bin/diamond /usr/bin/diamond; do
  command -v "$cand" >/dev/null 2>&1 && { DMND="$cand"; break; }
  [ -x "$cand" ] && { DMND="$cand"; break; }
done
if [ -z "$DMND" ]; then
  echo "!! DIAMOND not found. Reinstalling into ~/bin (no sudo needed):"
  cd ~ && curl -sL -o diamond.tar.gz \
    http://github.com/bbuchfink/diamond/releases/latest/download/diamond-linux64.tar.gz \
    && tar xzf diamond.tar.gz && mkdir -p ~/bin && mv -f diamond ~/bin/ && DMND="$HOME/bin/diamond"
fi
[ -z "$DMND" ] && { echo "!! still no DIAMOND, stopping"; exit 1; }
echo "diamond: $DMND  ($("$DMND" --version 2>&1 | head -1))"

# Make the PATH change stick for future shells, which is what failed last time.
grep -q 'HOME/bin' ~/.bashrc 2>/dev/null || echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc

[ -f ~/sprot.dmnd ] || { echo "!! ~/sprot.dmnd missing, run scripts/map_orfs.sh first"; exit 1; }
[ -f "$Q" ] || { echo "!! $Q missing, run scripts/16_build_twilight_refs.py first"; exit 1; }
echo "queries: $(grep -c '^>' "$Q") payload sequences"

if [ -s "$OUT" ]; then
  echo "$OUT already exists ($(wc -l < "$OUT") rows); delete it to redo"
else
  echo "searching with --ultra-sensitive (slower than default; this is the point)..."
  # --ultra-sensitive is not optional here. DIAMOND's default mode is tuned for close matches: measured
  # against exact Smith-Waterman on the same 16 payloads it found 922 homologs to parasail's 3,396, and
  # at <=35% identity, the regime this whole benchmark is about, 508 against 3,092, missing 87%.
  # GFP returned zero remote homologs in default mode and 255 by exact alignment. The looser e-value goes
  # with it; identity and coverage floors are applied afterwards in 16_build_twilight_refs.py anyway.
  "$DMND" blastp -q "$Q" -d ~/sprot --ultra-sensitive \
    --outfmt 6 qseqid sseqid pident length qlen slen bitscore evalue stitle \
    --max-target-seqs 1000 --evalue 10 --threads 4 --quiet -o "$OUT"
  echo "wrote $OUT ($(wc -l < "$OUT") hit rows)"
fi

echo
echo "=== DONE  $(date -Is) ==="
echo "Next, in Windows:"
echo "  python scripts\\16_build_twilight_refs.py --manifest data\\manifest.csv \\"
echo "    --manifest data\\manifest_auto.csv --manifest data\\synthetic\\synthetic_manifest.csv \\"
echo "    --swissprot \"%USERPROFILE%\\Downloads\\uniprot_sprot.fasta.gz\" \\"
echo "    --hits data\\reference\\payload_hits.tsv"
