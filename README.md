# Growing Neural Cellular Automata

A reimplementation of [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) (Mordvintsev, Randazzo, Niklasson, Levin; Distill 2020), the paper that grows a 🦎 from a single cell and regrows it after damage. Training is PyTorch on CUDA. Interaction is a native Rust app built on egui, which runs the trained rule on the CPU so you can erase part of the creature and watch it regenerate.

![gecko growing, regenerating, and growing rotated](eval/gecko_regenerating.png)

## Layout

| path | what it is |
|---|---|
| `SPEC.md` | The update rule and model file format both sides implement. |
| `train/nca.py` | The model: 16-channel state, Sobel perception, 48→128→16 MLP, stochastic updates, alive masking. |
| `train/train.py` | Trains one emoji under one regime (`growing`, `persistent`, `regenerating`). |
| `train/train_all.py` | Trains many (emoji, regime) pairs as concurrent processes. |
| `train/evaluate.py` | Scores growth, persistence, regeneration and rotation, and renders filmstrips to `eval/`. |
| `train/plot_losses.py` | Training curves to `eval/loss_curves.png`. |
| `train/hillclimb_round.sh` | Runs training variants across seeds for optimisation experiments. |
| `train/make_fixture.py` | Writes `fixtures/step_fixture.json`, the numeric contract the Rust engine is tested against. |
| `viewer/` | The egui app and its CPU engine. |
| `models/` | Trained models (`<emoji>_<regime>.json`); `.pt` checkpoints and logs are not committed. |

## The three regimes

The paper's three experiments, trained for each of its ten emojis (🦎 😀 💥 👁 🐠 🦋 🐞 🕸 🥨 🎄):

- **growing** trains from the seed alone. It grows the shape within ~96 steps but nothing tells it to stop, so it keeps growing.
- **persistent** samples starting states from a pool of its own past outputs, which teaches it to hold the shape as an attractor.
- **regenerating** is persistent plus random circular damage to the three best samples in each batch, which teaches it to repair.

## Running the viewer

```sh
cd viewer
cargo run --release                     # opens on gecko_regenerating, already growing
cargo run --release -- --models ../models
cargo run --release -- --bench ../models/gecko_regenerating.json --grid 72 --steps 3000
```

| input | action |
|---|---|
| left drag | erase a disc of cells (all 16 channels zeroed) |
| right click | plant a new seed |
| Space / S / R | play-pause / single step / reset to one seed |
| sliders | steps per frame, fire rate, rotation (rotates the perception kernels, as in the paper), brush radius |
| buttons | random damage (the trainer's circle damage), cut left half, grid size 72 to 256 |
| view menu | RGB over white, alpha, any hidden channel 4..15, hidden 4,5,6 as RGB |

The stats panel shows the step count, steps per second, live cells, and MSE against the target (the training loss, meaningful at rotation 0 on the 72 grid). `.cargo/config.toml` builds for the native CPU (AVX2/FMA). Delete it for a portable binary.

## Training

```sh
cd train
uv sync                                                    # torch cu128 (needs a Blackwell-capable build for the 5090)
uv run python train.py --emoji gecko --experiment regenerating
uv run python train_all.py --jobs 3                        # all 30 models
uv run python evaluate.py                                  # every model in ../models
uv run python plot_losses.py
uv run pytest -q
```

Hyperparameters follow the paper: Adam at 2e-3 dropping to 2e-4 after 2000 steps, 8000 steps, batch 8, a random 64 to 96 CA steps per iteration, per-parameter gradient normalisation, a pool of 1024, fire rate 0.5. `--overflow W` adds a penalty on states outside [-1, 1] (off by default; see Optimisation).

## Tests

- `cd viewer && cargo test --release` checks the Rust step against the PyTorch fixture (max abs error 2.4e-7 at angles 0 and 0.7), against a literal transcription of SPEC.md, and covers seeding and alive-masking edge cases.
- `cd train && uv run pytest -q` checks seeding, alive masking, target preprocessing and the JSON round trip.
