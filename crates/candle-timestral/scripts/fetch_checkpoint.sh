#!/usr/bin/env bash
# Download a Timestral checkpoint at a pinned Hub revision, for the golden tests.
#
# Usage: scripts/fetch_checkpoint.sh <repo> <revision> <dir>
#   e.g. scripts/fetch_checkpoint.sh Timestral-5M bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f /tmp/Timestral-5M
set -euo pipefail

if [ $# -ne 3 ]; then
    echo "usage: $0 <repo> <revision> <dir>" >&2
    exit 2
fi
repo=$1
rev=$2
dir=$3

mkdir -p "$dir"
for file in config.json model.safetensors; do
    curl -fsSL --retry 3 -o "$dir/$file" "https://huggingface.co/mistralai/$repo/resolve/$rev/$file"
done
