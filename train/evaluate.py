"""Score trained models on growth, persistence, regeneration and rotation; render filmstrips.

Every number is the mean pixel MSE of premultiplied RGBA against the target over a batch
of independent stochastic runs, so lower is better and values are comparable across models.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nca import load_json, load_target, make_seed, to_rgb

ROOT = Path(__file__).resolve().parent.parent
GROW_STEPS = 96
LONG_STEPS = (300, 1000, 2000)
REGEN_FROM = 300
REGEN_STEPS = 200
ANGLE = math.pi / 4


def damage_masks(h: int, w: int, device) -> dict[str, torch.Tensor]:
    ys = torch.arange(h, device=device)[:, None].float()
    xs = torch.arange(w, device=device)[None, :].float()
    cy, cx = h / 2, w / 2

    def keep(cut):
        return (~cut).float()[None, None]

    return {
        "circle_centre": keep((ys - cy) ** 2 + (xs - cx) ** 2 < 10**2),
        "circle_head": keep((ys - (cy - 12)) ** 2 + (xs - (cx - 10)) ** 2 < 9**2),
        "left_half": keep(xs.expand(h, w) < cx),
        "bottom_half": keep(ys.expand(h, w) > cy),
        "top_strip": keep((ys.expand(h, w) > cy - 14) & (ys.expand(h, w) < cy - 4)),
    }


def rotate(img: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate (N, C, H, W) counter-clockwise on screen by `angle` radians."""
    c, s = math.cos(angle), math.sin(angle)
    theta = torch.tensor([[c, s, 0.0], [-s, c, 0.0]], device=img.device)[None].expand(img.shape[0], 2, 3)
    grid = F.affine_grid(theta, img.shape, align_corners=False)
    return F.grid_sample(img, grid, align_corners=False)


def loss(x: torch.Tensor, target: torch.Tensor) -> float:
    return ((x[:, :4] - target) ** 2).mean().item()


def frame(x: torch.Tensor, scale: int = 3) -> np.ndarray:
    rgb = to_rgb(x[:1])[0].permute(1, 2, 0).cpu().numpy()
    img = (rgb * 255).astype(np.uint8)
    return img.repeat(scale, 0).repeat(scale, 1)


@torch.no_grad()
def evaluate(path: Path, n: int, out: Path, device) -> dict:
    ca, doc = load_json(path)
    ca = ca.to(device)
    target = load_target(doc["emoji"]).to(device)[None]
    h, w = target.shape[-2:]
    torch.manual_seed(1234)
    res = {"name": doc["name"], "experiment": doc["experiment"]}
    strip = []

    x = make_seed(n, h, w, device)
    for t in range(1, max(LONG_STEPS) + 1):
        x = ca(x)
        if t in (20, 40, 60, GROW_STEPS, *LONG_STEPS):
            strip.append(frame(x))
        if t == GROW_STEPS:
            res["loss_grow96"] = loss(x, target)
        if t in LONG_STEPS:
            res[f"loss_{t}"] = loss(x, target)
            res[f"alive_{t}"] = (x[:, 3] > 0.1).float().sum((1, 2)).mean().item()
        if t == REGEN_FROM:
            grown = x.clone()

    regen = {}
    for name, mask in damage_masks(h, w, device).items():
        y = grown * mask
        damaged = loss(y, target)
        for _ in range(REGEN_STEPS):
            y = ca(y)
        regen[name] = dict(damaged=damaged, recovered=loss(y, target))
        if name in ("circle_centre", "left_half"):
            strip += [frame(grown * mask), frame(y)]
    res["regen"] = regen
    res["loss_regen_mean"] = float(np.mean([v["recovered"] for v in regen.values()]))
    res["loss_damaged_mean"] = float(np.mean([v["damaged"] for v in regen.values()]))

    x = make_seed(n, h, w, device)
    for _ in range(REGEN_FROM):
        x = ca(x, angle=ANGLE)
    res["loss_rot45"] = loss(x, rotate(target, ANGLE))
    res["loss_rot45_wrong_sign"] = loss(x, rotate(target, -ANGLE))
    strip.append(frame(x))

    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(strip, 1)).save(out / f"{doc['name']}.png")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="*", type=Path)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--out", type=Path, default=ROOT / "eval")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = a.models or sorted((ROOT / "models").glob("*.json"))
    results = [evaluate(pth, a.n, a.out, device) for pth in paths]
    (a.out / "results.json").write_text(json.dumps(results, indent=1))
    cols = ["loss_grow96", "loss_300", "loss_2000", "alive_2000", "loss_damaged_mean", "loss_regen_mean", "loss_rot45"]
    print("| model | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for r in results:
        print(f"| {r['name']} | " + " | ".join(f"{r[c]:.5f}" if "loss" in c else f"{r[c]:.0f}" for c in cols) + " |")


if __name__ == "__main__":
    main()
