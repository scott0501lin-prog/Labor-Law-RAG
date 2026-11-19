#!/usr/bin/env bash
# Push the runtime files of the current commit to the Hugging Face Space.
#
#   git remote add space https://huggingface.co/spaces/<owner>/<space>   # once
#   bash scripts/deploy_hf.sh
#
# The Space gets a single snapshot commit, not our history: older commits hold
# non-LFS files over 10 MB (old text2vec index, judicial_cases.json), which
# HF rejects. Only what the container needs is included.
set -euo pipefail

REMOTE="${1:-space}"
FILES=(
  Dockerfile README.md requirements.txt .gitattributes .streamlit Fronted
  backend/embeddings/law_embeddings_bgem3.npy
  backend/embeddings/law_docs_bgem3.json
  backend/embeddings/cases_embeddings_bgem3.npy
  backend/embeddings/cases_docs_bgem3.json
)

git diff --quiet HEAD -- "${FILES[@]}" || { echo "Uncommitted changes in deploy files; commit first." >&2; exit 1; }
src=$(git rev-parse --short HEAD)

# Build the snapshot tree in a temporary index, leaving the working tree alone.
tmp_index=$(mktemp)
trap 'rm -f "$tmp_index"' EXIT
GIT_INDEX_FILE="$tmp_index" git read-tree --empty
GIT_INDEX_FILE="$tmp_index" git add -f "${FILES[@]}"
tree=$(GIT_INDEX_FILE="$tmp_index" git write-tree)
commit=$(git commit-tree "$tree" -m "deploy: snapshot of $(git rev-parse --abbrev-ref HEAD)@$src")

echo "Pushing $commit (from $src) to $REMOTE main ..."
git push --force "$REMOTE" "$commit:refs/heads/main"
