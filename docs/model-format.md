# Model format

The PyTorch trainer (`train/nca.py`) and the Rust viewer (`viewer/`) implement the same update rule. This file is the contract between them. `fixtures/step_fixture.json` pins it numerically.

## Model file (`models/<emoji>_<experiment>.json`)

| field | type | meaning |
|---|---|---|
| `format` | `"nca-v1"` | format tag |
| `name` | string | e.g. `gecko_regenerating` |
| `emoji` | string | the target emoji character |
| `emoji_name` | string | e.g. `gecko` |
| `experiment` | `growing` \| `persistent` \| `regenerating` | training regime |
| `channels` | 16 | state channels per cell: RGBA (premultiplied) then 12 hidden |
| `hidden` | 128 | hidden layer width |
| `fire_rate` | 0.5 | probability a cell updates in a step |
| `alive_threshold` | 0.1 | alpha threshold for liveness |
| `grid` | 72 | grid side the model was trained on |
| `w1` | f32[hidden * 48] | row-major `[hidden][48]` |
| `b1` | f32[hidden] | |
| `w2` | f32[channels * hidden] | row-major `[channels][hidden]`, no bias |
| `target_size` | 40 | |
| `target_rgba` | f32[40*40*4] | target image, row-major `[y][x][rgba]`, premultiplied alpha |
| `train` | object | training stats (free-form, informational) |

## State layout

The grid is `H x W` cells with 16 f32 channels each. Fixtures store it row-major `[y][x][c]`.

## One step

Inputs: state `s`, rotation angle `θ` (radians, 0 when not rotating), and a per-cell fire mask `m[y][x] ∈ {0,1}` drawn as `uniform(0,1) <= fire_rate`.

1. **Pre-alive.** `pre[y][x] = max over the 3x3 neighbourhood (in-bounds cells only) of s[.][.][3] > alive_threshold` (0.1 in every model so far).
2. **Perception.** Zero padding outside the grid. With `K = [[-1,0,1],[-2,0,2],[-1,0,1]] / 8` and `Kᵀ` its transpose,
   `kx = cosθ·K − sinθ·Kᵀ`, `ky = sinθ·K + cosθ·Kᵀ`.
   For each channel `c`, `gx_c(y,x) = Σ_{i,j∈{0,1,2}} kx[i][j] · s[y+i−1][x+j−1][c]` (cross-correlation, row `i` is the y offset), same for `gy` with `ky`.
   A positive `θ` rotates the grown pattern counter-clockwise on screen (y grows downward).
   Perception vector `p` (48): `p[0..16] = s[y][x][0..16]`, `p[16..32] = gx`, `p[32..48] = gy`.
3. **MLP.** `h = relu(w1 · p + b1)` (128), `ds = w2 · h` (16).
4. **Stochastic update.** `s' = s + ds · m[y][x]`.
5. **Post-alive.** `post` computed from `s'` exactly as in 1. This is the *unmasked* `s'`: a cell that step 6 zeroes can still keep its neighbour alive, so an implementation that skips dead cells must still update cells up to two away from life.
6. **Mask.** `s'[y][x][*] = 0` unless `pre[y][x] && post[y][x]`.

## Seed

All zeros except the centre cell `(H/2, W/2)` (integer division), whose channels `3..16` are `1.0`.

## Display

RGB over white: `rgb = clamp(1 − clamp(a,0,1) + s[0..3], 0, 1)` with `a = s[3]`.

## Fixture (`fixtures/step_fixture.json`)

`{ "model": <inline model object>, "cases": [ { "angle", "h", "w", "state_in", "fire_mask" (f32[h*w], 0/1), "state_out" } ] }`.
An implementation passes when every `state_out` element matches within `1e-4` absolute.
