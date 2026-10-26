"""Write fixtures/step_fixture.json: reference single steps the Rust engine must reproduce."""

import json
from pathlib import Path

import torch

from nca import CA, CHANNELS, export_json, make_seed

ROOT = Path(__file__).resolve().parent.parent


def main():
    torch.manual_seed(0)
    ca = CA()
    with torch.no_grad():
        ca.w2.weight.normal_(0, 0.05)
        ca.w1.bias.normal_(0, 0.1)
    tmp = ROOT / "fixtures" / "fixture_model.json"
    export_json(ca, tmp, dict(name="fixture", emoji="🦎", emoji_name="gecko", experiment="growing", train={}))
    model = json.loads(tmp.read_text())
    tmp.unlink()

    cases = []
    h, w = 24, 20
    x = make_seed(1, h, w)
    with torch.no_grad():
        for _ in range(12):
            x = ca(x, fire_rate=1.0)
        x += torch.randn_like(x) * 0.05 * (x[:, 3:4] > 0)
        for angle in (0.0, 0.7):
            mask = (torch.rand(1, 1, h, w) <= 0.5).float()
            out = ca(x, angle=angle, fire_mask=mask)
            cases.append(dict(
                angle=angle, h=h, w=w,
                state_in=x[0].permute(1, 2, 0).flatten().tolist(),
                fire_mask=mask.flatten().tolist(),
                state_out=out[0].permute(1, 2, 0).flatten().tolist(),
            ))
            alive = (out[0, 3] > 0.1).sum().item()
            print(f"angle={angle}: {alive} alive cells, |ds| max {(out - x).abs().max():.4f}")
    assert CHANNELS == 16
    (ROOT / "fixtures" / "step_fixture.json").write_text(json.dumps(dict(model=model, cases=cases)))


if __name__ == "__main__":
    main()
