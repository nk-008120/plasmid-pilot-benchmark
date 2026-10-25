#!/usr/bin/env bash
# Corpus setup: probe NCBI, fetch the GenBank synthetic division, install DIAMOND, map the 23 plasmids'
# ORFs to Swiss-Prot. Safe to re-run: every step is skipped if its output already exists.
#
#   bash /mnt/c/NK/plasmid-pilot/scripts/setup_corpus.sh


REPO=/mnt/c/NK/plasmid-pilot
LOG="$REPO/results/setup_corpus.log"
SPROT_WIN=/mnt/c/Users/pc/Downloads/uniprot_sprot.fasta.gz
GB=~/gbsyn
mkdir -p "$REPO/results" "$GB"
exec > >(tee "$LOG") 2>&1
echo "=== setup_corpus.sh  $(date -Is) ==="

say() { echo; echo "--- $* ---"; }

say "STEP 0: can this machine reach NCBI?"
curl -sS -o /dev/null -m 30 -w 'index: status %{http_code}  type %{content_type}  bytes %{size_download}\n' \
  https://ftp.ncbi.nlm.nih.gov/genbank/ || echo "index request FAILED"

say "STEP 1: probe for gbsyn files (listing-independent)"
: > "$GB/files.txt"
for i in $(seq 1 80); do
  code=$(curl -s -o /dev/null -m 30 -w '%{http_code}' -I "https://ftp.ncbi.nlm.nih.gov/genbank/gbsyn${i}.seq.gz")
  if [ "$code" = "200" ]; then
    echo "gbsyn${i}.seq.gz" >> "$GB/files.txt"
  else
    echo "gbsyn${i}.seq.gz -> $code  (stopping)"
    break
  fi
done
N=$(wc -l < "$GB/files.txt")
echo "found $N SYN files"
if [ "$N" -eq 0 ]; then
  echo "!! No SYN files reachable. Stopping before the download."
  echo "!! The fallback is Entrez E-utilities on a different host."
else
  say "STEP 2: download them (parallel, resumable)"
  cd "$GB"
  xargs -a files.txt -I{} -P4 curl -s -C - -O "https://ftp.ncbi.nlm.nih.gov/genbank/{}"
  ls -la "$GB"/*.seq.gz 2>/dev/null | tail -5
  echo "total: $(du -sh "$GB" | cut -f1)"
  say "STEP 2b: peek inside the first file (what are these records?)"
  zcat "$GB"/$(head -1 files.txt) 2>/dev/null | head -40
fi

say "STEP 3: DIAMOND"
if command -v diamond >/dev/null 2>&1; then
  echo "already installed: $(diamond --version 2>&1 | head -1)"
else
  cd ~ && curl -sL -o diamond.tar.gz \
    http://github.com/bbuchfink/diamond/releases/latest/download/diamond-linux64.tar.gz \
    && tar xzf diamond.tar.gz && mkdir -p ~/bin && mv diamond ~/bin/ && export PATH="$HOME/bin:$PATH"
  grep -q 'HOME/bin' ~/.bashrc 2>/dev/null || echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
  echo "installed: $(~/bin/diamond --version 2>&1 | head -1)"
fi
export PATH="$HOME/bin:$PATH"

say "STEP 4: Swiss-Prot DIAMOND database"
if [ -f ~/sprot.dmnd ]; then
  echo "already built: $(ls -la ~/sprot.dmnd | awk '{print $5" bytes"}')"
elif [ -f "$SPROT_WIN" ]; then
  diamond makedb --in "$SPROT_WIN" -d ~/sprot --quiet && echo "built ~/sprot.dmnd"
else
  echo "!! Swiss-Prot not found at $SPROT_WIN: adjust SPROT_WIN at the top of this script"
fi

say "STEP 5: map the 23 plasmids' ORFs to Swiss-Prot"
FAA="$REPO/results/baseline2/orfs.faa"
OUT="$REPO/results/baseline2/hits_23.tsv"
if [ ! -f "$FAA" ]; then
  echo "!! $FAA not found: run scripts/01_validate_pilot.py first"
elif [ ! -f ~/sprot.dmnd ]; then
  echo "!! no DIAMOND database, skipping"
else
  diamond blastp -q "$FAA" -d ~/sprot \
    --outfmt 6 qseqid sseqid pident length qlen slen bitscore evalue stitle \
    --max-target-seqs 25 --evalue 1e-5 --threads 4 --quiet -o "$OUT"
  echo "wrote $OUT  ($(wc -l < "$OUT") hit rows)"
  echo "first 3 rows:"; head -3 "$OUT"
fi

say "DONE  $(date -Is)"
echo "Next, in Windows:  python scripts\\12_autolabel.py --hits results\\baseline2\\hits_23.tsv --validate"
