use rayon::prelude::*;

use crate::model::{CHANNELS, HIDDEN, Model, PERCEPTION, Target};

const C: usize = CHANNELS;

/// `h x w` cells of `CHANNELS` f32 each, row-major `[y][x][c]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    pub h: usize,
    pub w: usize,
    pub state: Vec<f32>,
}

impl Grid {
    pub fn zeros(h: usize, w: usize) -> Self {
        Self { h, w, state: vec![0.0; h * w * C] }
    }

    pub fn seeded(h: usize, w: usize) -> Self {
        let mut g = Self::zeros(h, w);
        g.plant_seed(h / 2, w / 2);
        g
    }

    pub fn cell(&self, y: usize, x: usize) -> &[f32] {
        let i = (y * self.w + x) * C;
        &self.state[i..i + C]
    }

    pub fn plant_seed(&mut self, y: usize, x: usize) {
        let i = (y * self.w + x) * C;
        self.state[i + 3..i + C].fill(1.0);
    }

    pub fn clear_where(&mut self, mut pred: impl FnMut(usize, usize) -> bool) {
        let w = self.w;
        for (i, cell) in self.state.chunks_exact_mut(C).enumerate() {
            if pred(i / w, i % w) {
                cell.fill(0.0);
            }
        }
    }

    /// Clears cells whose centres lie within `r` of `(cy, cx)`, in continuous cell coordinates.
    pub fn erase_disc(&mut self, cy: f32, cx: f32, r: f32) {
        self.clear_where(|y, x| {
            let (dy, dx) = (y as f32 + 0.5 - cy, x as f32 + 0.5 - cx);
            dy * dy + dx * dx <= r * r
        });
    }

    /// The trainer's `circle_masks` damage: the grid spans `[-1, 1]` on both axes.
    pub fn erase_normalized_circle(&mut self, cy: f32, cx: f32, r: f32) {
        let (h, w) = (self.h, self.w);
        let coord = |i: usize, n: usize| -1.0 + 2.0 * i as f32 / (n.max(2) - 1) as f32;
        self.clear_where(|y, x| {
            let (dy, dx) = (coord(y, h) - cy, coord(x, w) - cx);
            dy * dy + dx * dx < r * r
        });
    }

    pub fn alive_count(&self, threshold: f32) -> usize {
        self.state.chunks_exact(C).filter(|c| c[3] > threshold).count()
    }

    /// The trainer's loss: mean squared RGBA error against the target centred on a zero background.
    pub fn mse(&self, target: &Target) -> f32 {
        let n = target.size as isize;
        let (oy, ox) = ((self.h as isize - n) / 2, (self.w as isize - n) / 2);
        let sum: f64 = (self.state.chunks_exact(C).enumerate())
            .map(|(i, cell)| {
                let (ty, tx) = ((i / self.w) as isize - oy, (i % self.w) as isize - ox);
                let t = if (0..n).contains(&ty) && (0..n).contains(&tx) {
                    let k = (ty * n + tx) as usize * 4;
                    &target.rgba[k..k + 4]
                } else {
                    &[0.0; 4]
                };
                cell[..4].iter().zip(t).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>()
            })
            .sum();
        (sum / (self.h * self.w * 4) as f64) as f32
    }
}

/// Per-cell masks reused across steps so stepping never allocates.
#[derive(Default)]
pub struct Scratch {
    alive: Vec<bool>,
    post: Vec<bool>,
}

struct Kernels {
    kx: [[f32; 3]; 3],
    ky: [[f32; 3]; 3],
}

impl Kernels {
    fn rotated(angle: f32) -> Self {
        let k = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]].map(|r: [f32; 3]| r.map(|v| v / 8.0));
        let (s, c) = angle.sin_cos();
        let mut kx = [[0.0; 3]; 3];
        let mut ky = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                kx[i][j] = c * k[i][j] - s * k[j][i];
                ky[i][j] = s * k[i][j] + c * k[j][i];
            }
        }
        Self { kx, ky }
    }
}

/// Whether any in-bounds cell within Chebyshev distance `r` of `(y, x)` is set.
fn any_near(mask: &[bool], h: usize, w: usize, y: usize, x: usize, r: usize) -> bool {
    let (y0, y1, x0, x1) = (y.saturating_sub(r), (y + r).min(h - 1), x.saturating_sub(r), (x + r).min(w - 1));
    (y0..=y1).any(|yy| mask[yy * w + x0..=yy * w + x1].iter().any(|&m| m))
}

fn perceive(src: &Grid, y: usize, x: usize, k: &Kernels) -> [f32; PERCEPTION] {
    let mut p = [0.0; PERCEPTION];
    p[..C].copy_from_slice(src.cell(y, x));
    for i in 0..3 {
        let Some(yy) = (y + i).checked_sub(1).filter(|&v| v < src.h) else { continue };
        for j in 0..3 {
            let Some(xx) = (x + j).checked_sub(1).filter(|&v| v < src.w) else { continue };
            let (a, b) = (k.kx[i][j], k.ky[i][j]);
            if a == 0.0 && b == 0.0 {
                continue;
            }
            let n = src.cell(yy, xx);
            for c in 0..C {
                p[C + c] += a * n[c];
                p[2 * C + c] += b * n[c];
            }
        }
    }
    p
}

fn mlp(model: &Model, p: &[f32; PERCEPTION]) -> [f32; C] {
    let mut h: [f32; HIDDEN] = model.b1;
    for (&v, w) in p.iter().zip(model.w1t.iter()) {
        if v != 0.0 {
            for (hj, wj) in h.iter_mut().zip(w) {
                *hj += v * wj;
            }
        }
    }
    let mut ds = [0.0; C];
    for (&hj, w) in h.iter().zip(model.w2t.iter()) {
        if hj > 0.0 {
            for (d, wc) in ds.iter_mut().zip(w) {
                *d += hj * wc;
            }
        }
    }
    ds
}

/// One update per docs/model-format.md. `fire` holds one entry per cell.
///
/// The MLP runs only for firing cells within two cells of an alive cell. Every other cell
/// ends up masked to zero, and its updated alpha cannot reach a pre-alive cell's post-alive test.
pub fn step(model: &Model, src: &Grid, dst: &mut Grid, angle: f32, fire: &[bool], scratch: &mut Scratch) {
    let (h, w) = (src.h, src.w);
    assert_eq!(fire.len(), h * w);
    if (dst.h, dst.w) != (h, w) {
        *dst = Grid::zeros(h, w);
    }
    let thr = model.alive_threshold;
    let Scratch { alive, post } = scratch;
    alive.clear();
    alive.extend(src.state.chunks_exact(C).map(|c| c[3] > thr));
    post.resize(h * w, false);

    let k = Kernels::rotated(angle);
    let alive: &[bool] = alive;
    dst.state
        .par_chunks_mut(w * C)
        .zip(post.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, (row, post_row))| {
            for (x, (out, post_out)) in row.chunks_exact_mut(C).zip(post_row).enumerate() {
                if !any_near(alive, h, w, y, x, 2) {
                    out.fill(0.0);
                    *post_out = false;
                    continue;
                }
                out.copy_from_slice(src.cell(y, x));
                if fire[y * w + x] {
                    let ds = mlp(model, &perceive(src, y, x, &k));
                    for (o, d) in out.iter_mut().zip(ds) {
                        *o += d;
                    }
                }
                *post_out = out[3] > thr;
            }
        });

    let post: &[bool] = post;
    dst.state.par_chunks_mut(w * C).enumerate().for_each(|(y, row)| {
        for (x, out) in row.chunks_exact_mut(C).enumerate() {
            if !(any_near(alive, h, w, y, x, 1) && any_near(post, h, w, y, x, 1)) {
                out.fill(0.0);
            }
        }
    });
}

/// A running simulation: double-buffered grid, scratch masks, and the fire-mask RNG.
pub struct Sim {
    pub grid: Grid,
    back: Grid,
    scratch: Scratch,
    fire: Vec<bool>,
    rng: fastrand::Rng,
    pub steps: u64,
}

impl Sim {
    pub fn new(size: usize) -> Self {
        Self {
            grid: Grid::seeded(size, size),
            back: Grid::zeros(size, size),
            scratch: Scratch::default(),
            fire: vec![false; size * size],
            rng: fastrand::Rng::new(),
            steps: 0,
        }
    }

    pub fn advance(&mut self, model: &Model, angle: f32, fire_rate: f32) {
        for f in &mut self.fire {
            *f = self.rng.f32() <= fire_rate;
        }
        step(model, &self.grid, &mut self.back, angle, &self.fire, &mut self.scratch);
        std::mem::swap(&mut self.grid, &mut self.back);
        self.steps += 1;
    }

    pub fn random_damage(&mut self) {
        let cy = self.rng.f32() - 0.5;
        let cx = self.rng.f32() - 0.5;
        let r = self.rng.f32() * 0.3 + 0.1;
        self.grid.erase_normalized_circle(cy, cx, r);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Case {
        angle: f32,
        h: usize,
        w: usize,
        state_in: Vec<f32>,
        fire_mask: Vec<f32>,
        state_out: Vec<f32>,
    }

    #[derive(Deserialize)]
    struct Fixture {
        model: serde_json::Value,
        cases: Vec<Case>,
    }

    fn fixture() -> (Model, Vec<Case>) {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../fixtures/step_fixture.json");
        let text = std::fs::read_to_string(path).expect("fixtures/step_fixture.json (run train/make_fixture.py)");
        let f: Fixture = serde_json::from_str(&text).unwrap();
        (Model::from_json(f.model).unwrap(), f.cases)
    }

    fn run(model: &Model, src: &Grid, angle: f32, fire: &[bool]) -> Grid {
        let mut out = Grid::zeros(src.h, src.w);
        step(model, src, &mut out, angle, fire, &mut Scratch::default());
        out
    }

    /// docs/model-format.md transcribed literally, with no skipping, to check `step` against.
    fn reference_step(model: &Model, s: &Grid, angle: f32, fire: &[bool]) -> Grid {
        let (h, w, thr) = (s.h, s.w, model.alive_threshold);
        let alive_of = |g: &Grid| -> Vec<bool> {
            let a: Vec<bool> = g.state.chunks_exact(C).map(|c| c[3] > thr).collect();
            (0..h * w).map(|i| any_near(&a, h, w, i / w, i % w, 1)).collect()
        };
        let pre = alive_of(s);
        let k = Kernels::rotated(angle);
        let mut out = s.clone();
        for y in 0..h {
            for x in 0..w {
                let p = perceive(s, y, x, &k);
                let mut hid = [0.0f32; HIDDEN];
                for (j, hj) in hid.iter_mut().enumerate() {
                    let z: f32 = model.b1[j] + (0..PERCEPTION).map(|i| model.w1t[i][j] * p[i]).sum::<f32>();
                    *hj = z.max(0.0);
                }
                let m = if fire[y * w + x] { 1.0 } else { 0.0 };
                for c in 0..C {
                    let ds: f32 = (0..HIDDEN).map(|j| model.w2t[j][c] * hid[j]).sum();
                    out.state[(y * w + x) * C + c] += ds * m;
                }
            }
        }
        let post = alive_of(&out);
        out.clear_where(|y, x| !(pre[y * w + x] && post[y * w + x]));
        out
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    #[test]
    fn matches_pytorch_fixture() {
        let (model, cases) = fixture();
        assert_eq!(cases.len(), 2);
        for case in cases {
            let src = Grid { h: case.h, w: case.w, state: case.state_in };
            let fire: Vec<bool> = case.fire_mask.iter().map(|&m| m != 0.0).collect();
            let out = run(&model, &src, case.angle, &fire);
            let err = max_abs_diff(&out.state, &case.state_out);
            eprintln!("angle {}: max abs error {err:e}", case.angle);
            assert!(err < 1e-4, "angle {}: max abs error {err}", case.angle);
        }
    }

    #[test]
    fn matches_literal_spec_on_noisy_state() {
        let (model, _) = fixture();
        let mut rng = fastrand::Rng::with_seed(7);
        let (h, w) = (20, 23);
        let mut src = Grid::zeros(h, w);
        for v in &mut src.state {
            *v = (rng.f32() - 0.5) * 0.4;
        }
        let fire: Vec<bool> = (0..h * w).map(|_| rng.bool()).collect();
        for angle in [0.0, 0.7, -2.0] {
            let fast = run(&model, &src, angle, &fire);
            let reference = reference_step(&model, &src, angle, &fire);
            assert!(max_abs_diff(&fast.state, &reference.state) < 1e-5);
        }
    }

    /// A dead-neighbourhood cell's own update is discarded, yet its new alpha can keep a neighbour alive.
    #[test]
    fn dead_cell_update_can_keep_a_neighbour_alive() {
        let (mut model, _) = fixture();
        *model.w1t = [[0.0; HIDDEN]; PERCEPTION];
        *model.w2t = [[0.0; C]; HIDDEN];
        model.b1 = [0.0; HIDDEN];
        (model.w1t[3][0], model.b1[0], model.w2t[0][3]) = (10.0, -1.5, -1.0);
        (model.w1t[3][1], model.b1[1], model.w2t[1][3]) = (-10.0, 1.0, 0.5);

        let mut src = Grid::zeros(1, 4);
        src.state[3] = 0.2;
        src.state[C + 5] = 0.3;
        let fire = [true, false, true, false];
        let out = run(&model, &src, 0.0, &fire);
        assert_eq!(out.cell(0, 1)[5], 0.3, "neighbour kept alive by the dead cell's new alpha");
        assert_eq!(out, reference_step(&model, &src, 0.0, &fire));
    }

    #[test]
    fn seed_is_placed_at_centre() {
        for (h, w) in [(72, 72), (24, 20), (7, 9)] {
            let g = Grid::seeded(h, w);
            for y in 0..h {
                for x in 0..w {
                    let expected: Vec<f32> = if (y, x) == (h / 2, w / 2) {
                        [0.0; 3].into_iter().chain([1.0; C - 3]).collect()
                    } else {
                        vec![0.0; C]
                    };
                    assert_eq!(g.cell(y, x), expected, "cell ({y}, {x}) of {h}x{w}");
                }
            }
        }
    }

    #[test]
    fn cells_without_live_neighbours_stay_zero() {
        let (model, _) = fixture();
        let (h, w) = (16, 16);
        let mut src = Grid::seeded(h, w);
        let i = (2 * w + 2) * C;
        src.state[i..i + C].fill(0.09);
        let out = run(&model, &src, 0.0, &vec![true; h * w]);
        for y in 0..h {
            for x in 0..w {
                let near_seed = y.abs_diff(h / 2) <= 1 && x.abs_diff(w / 2) <= 1;
                if !near_seed {
                    assert!(out.cell(y, x).iter().all(|&v| v == 0.0), "cell ({y}, {x}) should be dead");
                }
            }
        }
    }
}
