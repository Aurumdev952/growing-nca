import math

import torch

from nca import CA, CHANNELS, export_json, load_json, load_target, make_seed


def test_seed_is_single_centre_cell():
    x = make_seed(1, 9, 8)
    alive = (x[0, 3] > 0).nonzero().tolist()
    assert alive == [[4, 4]]
    assert x[0, :3].abs().sum() == 0 and torch.all(x[0, 3:, 4, 4] == 1)


def test_cells_far_from_life_stay_dead():
    torch.manual_seed(0)
    ca = CA()
    torch.nn.init.normal_(ca.w2.weight, 0, 0.5)
    torch.nn.init.normal_(ca.w1.bias, 0, 0.5)
    x = ca(make_seed(1, 16, 16), fire_rate=1.0)
    assert x[0, :, :6].abs().sum() == 0 and x[0, :, :, :6].abs().sum() == 0


def test_untrained_model_is_identity():
    x = make_seed(2, 12, 12)
    assert torch.equal(CA()(x), x)


def test_target_is_premultiplied_and_padded():
    t = load_target("🦎")
    assert t.shape == (4, 72, 72)
    assert t[:, :16].abs().sum() == 0
    assert torch.all(t[:3] <= t[3:4] + 1e-6)


def test_json_roundtrip_preserves_dynamics(tmp_path):
    torch.manual_seed(1)
    ca = CA()
    torch.nn.init.normal_(ca.w2.weight, 0, 0.05)
    meta = dict(name="t", emoji="🦎", emoji_name="gecko", experiment="growing", train={})
    export_json(ca, tmp_path / "m.json", meta)
    ca2, doc = load_json(tmp_path / "m.json")
    assert doc["channels"] == CHANNELS and len(doc["target_rgba"]) == 40 * 40 * 4
    x = make_seed(1, 20, 20)
    mask = torch.ones(1, 1, 20, 20)
    for angle in (0.0, math.pi / 3):
        assert torch.allclose(ca(x, angle=angle, fire_mask=mask), ca2(x, angle=angle, fire_mask=mask), atol=1e-6)
