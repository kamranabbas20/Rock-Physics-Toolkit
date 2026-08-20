#!/usr/bin/env bash
# Build a wheelhouse so the toolkit can be installed on a machine with no
# network at all.
#
# Run this ONCE on a machine that does have a network, then carry the whole
# repository (wheelhouse included) across.
#
#   ./scripts/make_offline_bundle.sh                  # for this machine
#   ./scripts/make_offline_bundle.sh win_amd64 3.11   # cross-build for Windows
#
# On the air-gapped machine:
#   python3 -m venv .venv
#   .venv/bin/pip install --no-index --find-links wheelhouse -r requirements-lock.txt
#   .venv/bin/streamlit run avo_qi/app.py
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

platform="${1:-}"
pyversion="${2:-}"
out="wheelhouse"

mkdir -p "$out"
args=(download --dest "$out" --requirement requirements-lock.txt)

if [[ -n "$platform" ]]; then
  # A cross-platform download cannot build from source, so every dependency
  # must have a wheel for the target. All of ours do.
  args+=(--platform "$platform" --only-binary=:all:)
  [[ -n "$pyversion" ]] && args+=(--python-version "$pyversion")
  echo "Downloading wheels for platform=$platform python=$pyversion"
else
  echo "Downloading wheels for this machine"
fi

python3 -m pip "${args[@]}"

count=$(find "$out" -name '*.whl' -o -name '*.tar.gz' | wc -l | tr -d ' ')
size=$(du -sh "$out" | cut -f1)
echo
echo "Wheelhouse ready: $count files, $size in $out/"
echo "Copy the repository across, then on the offline machine run:"
echo "  python3 -m venv .venv"
echo "  .venv/bin/pip install --no-index --find-links $out -r requirements-lock.txt"
