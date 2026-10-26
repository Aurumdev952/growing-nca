"""Growing Neural Cellular Automata (Mordvintsev et al., Distill 2020).

The update rule must stay bit-for-bit compatible with SPEC.md, because the Rust
viewer re-implements it from the exported JSON.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

CHANNELS = 16
HIDDEN = 128
TARGET_SIZE = 40
TARGET_PAD = 16
ALIVE_THRESHOLD = 0.1
EMOJI_DIR = Path(__file__).parent / "emoji"

EMOJIS = {
    "gecko": "🦎",
    "smiley": "😀",
    "boom": "💥",
    "eye": "👁",
    "fish": "🐠",
    "butterfly": "🦋",
    "ladybug": "🐞",
    "spiderweb": "🕸",
    "pretzel": "🥨",
    "tree": "🎄",
}

EXPERIMENTS = {
    "growing": dict(use_pool=False, damage_n=0),
    "persistent": dict(use_pool=True, damage_n=0),
    "regenerating": dict(use_pool=True, damage_n=3),
}


def emoji_codepoint(emoji: str) -> str:
    return "_".join(f"{ord(c):x}" for c in emoji if ord(c) != 0xFE0F)


def load_target(emoji: str) -> torch.Tensor:
    """Premultiplied RGBA, shape (4, 72, 72), values in [0, 1]."""
    img = Image.open(EMOJI_DIR / f"emoji_u{emoji_codepoint(emoji)}.png").convert("RGBA")
    img.thumbnail((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32) / 255.0
    a[..., :3] *= a[..., 3:4]
    t = torch.from_numpy(a).permute(2, 0, 1)
    return F.pad(t, (TARGET_PAD,) * 4)


def make_seed(n: int, h: int, w: int, device=None) -> torch.Tensor:
    x = torch.zeros(n, CHANNELS, h, w, device=device)
    x[:, 3:, h // 2, w // 2] = 1.0
    return x


def to_rgb(x: torch.Tensor) -> torch.Tensor:
    """Composite premultiplied RGBA over white."""
    a = x[:, 3:4].clamp(0, 1)
    return (1.0 - a + x[:, :3]).clamp(0, 1)


def alive_mask(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x[:, 3:4], 3, stride=1, padding=1) > ALIVE_THRESHOLD


class CA(nn.Module):
    def __init__(self, fire_rate: float = 0.5):
        super().__init__()
        self.fire_rate = fire_rate
        self.w1 = nn.Conv2d(3 * CHANNELS, HIDDEN, 1)
        self.w2 = nn.Conv2d(HIDDEN, CHANNELS, 1, bias=False)
        nn.init.zeros_(self.w2.weight)
        sobel = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        self.register_buffer("sobel_x", sobel)
        self.register_buffer("sobel_y", sobel.T.contiguous())

    def perceive(self, x: torch.Tensor, angle: float) -> torch.Tensor:
        c, s = math.cos(angle), math.sin(angle)
        kx = c * self.sobel_x - s * self.sobel_y
        ky = s * self.sobel_x + c * self.sobel_y
        ch = x.shape[1]
        gx = F.conv2d(x, kx.expand(ch, 1, 3, 3), padding=1, groups=ch)
        gy = F.conv2d(x, ky.expand(ch, 1, 3, 3), padding=1, groups=ch)
        return torch.cat([x, gx, gy], 1)

    def forward(self, x, fire_rate=None, angle=0.0, fire_mask=None):
        pre = alive_mask(x)
        dx = self.w2(F.relu(self.w1(self.perceive(x, angle))))
        if fire_mask is None:
            rate = self.fire_rate if fire_rate is None else fire_rate
            fire_mask = torch.rand_like(x[:, :1]) <= rate
        x = x + dx * fire_mask
        return x * (pre & alive_mask(x))


def export_json(ca: CA, path: Path, meta: dict) -> None:
    w1 = ca.w1.weight.detach().cpu().reshape(HIDDEN, 3 * CHANNELS)
    b1 = ca.w1.bias.detach().cpu()
    w2 = ca.w2.weight.detach().cpu().reshape(CHANNELS, HIDDEN)
    target = load_target(meta["emoji"])[:, TARGET_PAD:-TARGET_PAD, TARGET_PAD:-TARGET_PAD]
    doc = {
        "format": "nca-v1",
        **meta,
        "channels": CHANNELS,
        "hidden": HIDDEN,
        "fire_rate": ca.fire_rate,
        "alive_threshold": ALIVE_THRESHOLD,
        "grid": TARGET_SIZE + 2 * TARGET_PAD,
        "w1": w1.flatten().tolist(),
        "b1": b1.tolist(),
        "w2": w2.flatten().tolist(),
        "target_size": TARGET_SIZE,
        "target_rgba": target.permute(1, 2, 0).flatten().tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


def load_json(path: Path) -> tuple[CA, dict]:
    doc = json.loads(Path(path).read_text())
    ca = CA(doc["fire_rate"])
    with torch.no_grad():
        ca.w1.weight.copy_(torch.tensor(doc["w1"]).reshape(HIDDEN, 3 * CHANNELS, 1, 1))
        ca.w1.bias.copy_(torch.tensor(doc["b1"]))
        ca.w2.weight.copy_(torch.tensor(doc["w2"]).reshape(CHANNELS, HIDDEN, 1, 1))
    return ca, doc
