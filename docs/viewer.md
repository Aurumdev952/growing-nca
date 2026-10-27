# Viewer

The viewer is a native egui app in `viewer/`. It loads every `models/*.json` file and runs the update rule on the CPU across all cores with rayon.

## Running

```sh
cd viewer
cargo run --release                                    # loads ../models
cargo run --release -- --models path/to/models         # another model folder
```

## Controls

| input | action |
|---|---|
| left drag | Erase a disc of cells. All 16 channels are set to zero. |
| right click | Plant a new seed. |
| Space | Play or pause. |
| S | Advance one step. |
| R | Reset to a single seed. |

The side panel has these sections:

- **Models.** Pick an emoji, then one of its regimes (growing, persistent, regenerating). Switching emoji keeps the current regime. Rescan reloads the model folder.
- **Simulation.** Steps per frame, fire rate (the chance each cell updates in a step), and rotation. Rotation turns the perception kernels as in the paper, so a model grows its shape at an angle it never saw in training. Grid size runs from 72 to 256.
- **Damage.** Brush radius, random damage (the same circles the trainer uses), and cutting off the left half.
- **View.** RGB over white, alpha, any hidden channel from 4 to 15, or hidden channels 4, 5 and 6 shown as RGB.
- **Stats.** Step count, steps per second, cost per step, live cells, and MSE against the target. MSE is the training loss, so it means most at rotation 0 on the 72 grid.

## Benchmarking

```sh
cargo run --release -- --bench ../models/gecko_regenerating.json --grid 72 --steps 3000
```

This reports steps per second from the seed and continuing from there. On a 72x72 grid the median of 5 runs was 2437 steps per second, ranging from 2169 to 2785, measured while training ran on the same machine. A 256x256 grid ran at about 570 steps per second.

The engine runs the MLP only for cells within two cells of a live one. That is the widest region the update can change, as [model-format.md](model-format.md) explains.

## Portability

`viewer/.cargo/config.toml` compiles for the CPU you build on (`target-cpu=native`). Building from source works anywhere. To build a binary that runs on other machines, delete that file first.

## Tests

```sh
cargo test --release
```

The tests check the Rust step against a PyTorch fixture (largest absolute error 2.4e-7 at angles 0 and 0.7), against a literal transcription of the spec, and cover seeding and the alive-mask edge cases.

`viewer/scripts/drive_xvfb.sh <dir>` runs the app on its own Xvfb display. It grows the gecko, pauses, erases, regrows and switches model, and saves a screenshot after each step. It never sends input to your desktop. It needs `Xvfb`, `xdotool` and `ffmpeg`.
