#!/usr/bin/env bash
# DIAMOND setup and ORF->Swiss-Prot mapping, for the 23 audited plasmids and for the NCBI corpus.
# Idempotent: every step is skipped if its output already exists, so re-run it freely.
#
#   bash /mnt/c/NK/plasmid-pilot/scripts/setup_corpus.sh
#
# Logs to results/setup_corpus.log inside the repo, which Claude reads directly: nothing to copy back.

REPO=/mnt/c/NK/plasmid-pilot
LOG="$REPO/results/setup_corpus.log"
SPROT_WIN=/mnt/c/Users/pc/Downloads/uniprot_sprot.fasta.gz
mkdir -p "$REPO/results"
exec > >(tee "$LOG") 2>&1
echo "=== setup_corpus.sh  $(date -Is) ==="
say() { echo; echo "--- $* ---"; }
export PATH="$HOME/bin:$PATH"

say "STEP 1: DIAMOND"
if command -v diamond >/dev/null 2>&1; then
  echo "already installed: $(diamond --version 2>&1 | head -1)"
else
  cd ~ && curl -sL -o diamond.tar.gz \
    http://github.com/bbuchfink/diamond/releases/latest/download/diamond-linux64.tar.gz \
    && tar xzf diamond.tar.gz && mkdir -p ~/bin && mv diamond ~/bin/
  grep -q 'HOME/bin' ~/.bashrc 2>/dev/null || echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
  echo "installed: $(~/bin/diamond --version 2>&1 | head -1)"
fi
command -v diamond >/dev/null 2>&1 || { echo "!! DIAMOND unavailable, stopping"; exit 1; }

say "STEP 2: Swiss-Prot DIAMOND database"
if [ -f ~/sprot.dmnd ]; then
  echo "already built: $(du -h ~/sprot.dmnd | cut -f1)"
elif [ -f "$SPROT_WIN" ]; then
  diamond makedb --in "$SPROT_WIN" -d ~/sprot --quiet && echo "built ~/sprot.dmnd ($(du -h ~/sprot.dmnd | cut -f1))"
else
  echo "!! Swiss-Prot not found at $SPROT_WIN: edit SPROT_WIN at the top of this script"; exit 1
fi

# $1 = ORF fasta, $2 = output tsv, $3 = label
map_orfs() {
  if [ ! -f "$1" ]; then
    echo "   $3: $1 not found, skipping"
  elif [ -s "$2" ]; then
    echo "   $3: $2 already exists ($(wc -l < "$2") rows); delete it to redo"
  else
    echo " $3: mapping $(grep -c '^>' "$1") ORFs..."
    diamond blastp -q "$1" -d ~/sprot \
      --outfmt 6 qseqid sseqid pident length qlen slen bitscore evalue stitle \
      --max-target-seqs 25 --evalue 1e-5 --threads 4 --quiet -o "$2"
    echo "   $3: wrote $2 ($(wc -l < "$2") hit rows)"
  fi
}

say "STEP 3: map ORFs to Swiss-Prot"
map_orfs "$REPO/results/baseline2/orfs.faa" "$REPO/results/baseline2/hits_23.tsv" "23 audited plasmids"
map_orfs "$REPO/results/corpus/orfs.faa"    "$REPO/results/corpus/hits_corpus.tsv" "NCBI corpus"

say "DONE  $(date -Is)"
echo "Next, in Windows:"
echo "  python scripts\\12_autolabel.py --hits results\\baseline2\\hits_23.tsv --validate"
