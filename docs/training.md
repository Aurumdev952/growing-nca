# Training

The trainer lives in `train/`. It is PyTorch with `torch.compile`, managed with uv. `pyproject.toml` pulls torch from the CUDA 12.8 wheel index, which Blackwell GPUs such as the RTX 5090 need.

```sh
cd train
uv sync
```

## The model

Each cell holds 16 channels: premultiplied RGBA, then 12 hidden channels. One step works like this:

1. Each cell perceives itself and the Sobel x and y gradients of its 3x3 neighbourhood, giving 48 numbers.
2. A 48 to 128 to 16 MLP turns those into an update. The last layer starts at zero, so a new model does nothing.
3. Each cell applies its update with probability 0.5, so there is no global clock.
4. A cell stays alive only if some cell in its 3x3 neighbourhood has alpha above 0.1, both before and after the update.

Growth starts from one live cell in the centre of a 72x72 grid. Targets are 40x40 emojis padded by 16 cells on each side. [model-format.md](model-format.md) defines the rule exactly.

## The three regimes

The paper's three experiments, available for each emoji:

- **growing** starts every sample from the seed. It grows the shape within about 96 steps, but nothing teaches it to stop, so it keeps growing.
- **persistent** draws starting states from a pool of 1024 of its own past outputs, and replaces the worst one with a fresh seed. This teaches it to hold the shape.
- **regenerating** is persistent plus circular damage to the three best samples in every batch. This teaches it to repair.

## Commands

```sh
uv run python train.py --emoji gecko --experiment regenerating   # one model
uv run python train_all.py --jobs 3                              # all 30 models
uv run python evaluate.py                                        # score every model in ../models
uv run python plot_losses.py                                     # training curves
uv run pytest -q                                                 # tests
```

| flag (`train.py`) | default | meaning |
|---|---|---|
| `--emoji` | `gecko` | One of gecko, smiley, boom, eye, fish, butterfly, ladybug, spiderweb, pretzel, tree. |
| `--experiment` | `regenerating` | `growing`, `persistent` or `regenerating`. |
| `--steps` | 8000 | Training iterations. |
| `--batch` | 8 | Samples per iteration. |
| `--pool` | 1024 | Pool size for the persistent and regenerating regimes. |
| `--lr` | 2e-3 | Adam learning rate, divided by 10 after 2000 iterations. |
| `--seed` | 0 | RNG seed. |
| `--overflow` | 0 | Weight of a penalty on states outside [-1, 1]. Off by default. |
| `--no-compile` | | Run eagerly without `torch.compile`. |
| `--out` | `../models` | Output folder. |

Each iteration runs a random 64 to 96 CA steps and normalises each parameter's gradient, as in the paper. A run writes `<emoji>_<regime>.json` (the viewer's format), a `.pt` checkpoint, and a loss history under `runs/`.

On a power-limited laptop GPU, three parallel jobs are as fast as more. [optimisation.md](optimisation.md) explains why.

## Evaluation

`evaluate.py` runs each model 16 times with a fixed RNG seed and records:

- MSE against the target at steps 96, 300, 1000 and 2000;
- live cells at step 2000;
- MSE right after five fixed damage patterns applied at step 300, and again 200 steps later;
- MSE when grown with the kernels rotated 45°.

It writes `results.json`, a markdown table and a filmstrip per model to `eval/`. Use `--n` and `--seed` to re-check a result on fresh rollouts. [results.md](results.md) has the numbers.

## Picking the best seed

Seed-to-seed variance is large. `hillclimb_round.sh` trains variants across several seeds into `models/hillclimb/`. `select_seeds.sh <emoji>` scores them on both standard and held-out rollouts. `promote.py` then copies the best one into `models/`, ranking on MSE at step 2000 plus mean MSE after damage.

## Tests

- `uv run pytest -q` checks seeding, alive masking, target preprocessing and the JSON round trip.
- `make_fixture.py` writes `fixtures/step_fixture.json`, the numeric reference the Rust engine is tested against.
