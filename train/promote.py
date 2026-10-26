"""Copy the best evaluated candidate into models/ under its canonical name.

Score = loss_2000 + loss_regen_mean: the regenerating regime's two goals, holding the
shape long after training's horizon and repairing it after damage.
"""

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def score(r: dict) -> float:
    return r["loss_2000"] + r["loss_regen_mean"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path, help="evaluate.py results.json covering one emoji's candidates")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    cands = json.loads(a.results.read_text())
    for r in sorted(cands, key=score):
        print(f"{r['name']:20s} score {score(r):.5f}  loss_2000 {r['loss_2000']:.5f}  regen {r['loss_regen_mean']:.5f}")
    best = min(cands, key=score)
    src = next((ROOT / "models" / "hillclimb" / best["name"]).glob("*_regenerating.json"))
    dst = ROOT / "models" / src.name
    print(f"promote {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
    if not a.dry_run:
        shutil.copy(src, dst)
        shutil.copy(src.with_suffix(".pt"), dst.with_suffix(".pt"))


if __name__ == "__main__":
    main()
