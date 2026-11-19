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

git diff --quiet HEAD -- "${FILES[@]}" || echo "Note: uncommitted changes are NOT deployed (snapshot uses HEAD)." >&2
src=$(git rev-parse --short HEAD)

# Build the snapshot tree from HEAD's committed blobs (LFS files stay as
# pointers) in a temporary index, leaving the working tree alone.
tmp_index=$(mktemp)
trap 'rm -f "$tmp_index"' EXIT
GIT_INDEX_FILE="$tmp_index" git read-tree --empty
git ls-tree -r HEAD -- "${FILES[@]}" | GIT_INDEX_FILE="$tmp_index" git update-index --index-info
tree=$(GIT_INDEX_FILE="$tmp_index" git write-tree)
commit=$(git commit-tree "$tree" -m "deploy: snapshot of $(git rev-parse --abbrev-ref HEAD)@$src")

echo "Pushing $commit (from $src) to $REMOTE main ..."
git push --force "$REMOTE" "$commit:refs/heads/main"
