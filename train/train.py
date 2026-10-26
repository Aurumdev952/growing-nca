"""Train one NCA on one emoji under one experiment regime (growing / persistent / regenerating)."""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from nca import CA, EMOJIS, EXPERIMENTS, export_json, load_target, make_seed

ROOT = Path(__file__).resolve().parent.parent


def circle_masks(n: int, h: int, w: int, device) -> torch.Tensor:
    """1 outside a random circle, 0 inside; centre in [-0.5, 0.5], radius in [0.1, 0.4] (grid spans [-1, 1])."""
    ys = torch.linspace(-1, 1, h, device=device)[None, :, None]
    xs = torch.linspace(-1, 1, w, device=device)[None, None, :]
    centre = torch.rand(2, n, 1, 1, device=device) - 0.5
    r = torch.rand(n, 1, 1, device=device) * 0.3 + 0.1
    inside = ((xs - centre[0]) ** 2 + (ys - centre[1]) ** 2) < r**2
    return (~inside).float()[:, None]


def per_sample_loss(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((x[:, :4] - target) ** 2).mean(dim=(1, 2, 3))


def train(emoji_name: str, experiment: str, steps: int, batch: int, pool_size: int,
          lr: float, seed: int, compile_step: bool, out: Path, log_every: int) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    cfg = EXPERIMENTS[experiment]
    emoji = EMOJIS[emoji_name]

    target = load_target(emoji).to(device)[None]
    h, w = target.shape[-2:]
    seed_state = make_seed(1, h, w, device)
    ca = CA().to(device)
    opt = torch.optim.Adam(ca.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[2000], gamma=0.1)
    step_fn = torch.compile(ca) if compile_step else ca
    pool = seed_state.repeat(pool_size, 1, 1, 1)

    history = []
    t0 = time.time()
    for it in range(steps):
        if cfg["use_pool"]:
            idx = torch.randperm(pool_size, device=device)[:batch]
            x0 = pool[idx]
            with torch.no_grad():
                order = per_sample_loss(x0, target).argsort(descending=True)
            idx, x0 = idx[order], x0[order]
            x0[:1] = seed_state
            if cfg["damage_n"]:
                x0[-cfg["damage_n"]:] *= circle_masks(cfg["damage_n"], h, w, device)
        else:
            x0 = seed_state.repeat(batch, 1, 1, 1)

        x = x0
        for _ in range(int(torch.randint(64, 97, ()).item())):
            x = step_fn(x)
        loss = per_sample_loss(x, target).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            for p in ca.parameters():
                p.grad /= p.grad.norm() + 1e-8
        opt.step()
        sched.step()
        if cfg["use_pool"]:
            pool[idx] = x.detach()

        history.append(loss.detach())
        if it % log_every == 0 or it == steps - 1:
            print(f"[{emoji_name}/{experiment}] step {it:5d} loss {loss.item():.6f} "
                  f"log10 {torch.log10(loss).item():.3f} {time.time() - t0:.0f}s", flush=True)

    history = torch.stack(history).tolist()

    stats = dict(steps=steps, batch=batch, pool_size=pool_size, lr=lr, seed=seed,
                 final_loss=sum(history[-100:]) / min(100, len(history)),
                 seconds=round(time.time() - t0, 1))
    name = f"{emoji_name}_{experiment}"
    meta = dict(name=name, emoji=emoji, emoji_name=emoji_name, experiment=experiment, train=stats)
    export_json(ca, out / f"{name}.json", meta)
    torch.save(ca.state_dict(), out / f"{name}.pt")
    (out / f"{name}.loss.json").write_text(json.dumps(history))
    print(f"[{name}] done: {stats}", flush=True)
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emoji", default="gecko", choices=EMOJIS)
    p.add_argument("--experiment", default="regenerating", choices=EXPERIMENTS)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--pool", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--out", type=Path, default=ROOT / "models")
    p.add_argument("--log-every", type=int, default=200)
    a = p.parse_args()
    train(a.emoji, a.experiment, a.steps, a.batch, a.pool, a.lr, a.seed, a.compile, a.out, a.log_every)


if __name__ == "__main__":
    main()
