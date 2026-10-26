#!/usr/bin/env bash
# Usage: hillclimb_round.sh "<variant> <seed> <train.py args...>" ...  (3 runs at a time; GPU is power-bound)
cd "$(dirname "$0")"
for spec in "$@"; do echo "$spec"; done | xargs -P 3 -I{} bash -c '
  set -- {}; variant=$1; seed=$2; shift 2
  out=../models/hillclimb/${variant}_s${seed}
  mkdir -p "$out/runs"
  uv run python train.py --emoji gecko --experiment regenerating --seed "$seed" --out "$out" "$@" > "$out/runs/train.log" 2>&1 \
    && echo "ok ${variant}_s${seed}" || echo "FAILED ${variant}_s${seed}"'
