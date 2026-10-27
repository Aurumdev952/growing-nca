#!/usr/bin/env bash
# Score <emoji>_base_s{0,1,2} on the standard rollouts and on held-out ones, then promote the held-out winner.
# usage: select_seeds.sh <emoji>...
set -euo pipefail
cd "$(dirname "$0")"
for e in "$@"; do
  models=(../models/hillclimb/"${e}"_base_s*/"${e}"_regenerating.json)
  uv run python evaluate.py "${models[@]}" --out "../eval/hillclimb/$e" | tail -n +3
  uv run python evaluate.py "${models[@]}" --n 32 --seed 99 --out "../eval/hillclimb/${e}_heldout" > /dev/null
  uv run python promote.py "../eval/hillclimb/${e}_heldout/results.json"
done
