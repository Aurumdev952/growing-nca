use std::path::Path;

use serde::Deserialize;

pub const CHANNELS: usize = 16;
pub const HIDDEN: usize = 128;
pub const PERCEPTION: usize = 3 * CHANNELS;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Experiment {
    Growing,
    Persistent,
    Regenerating,
}

impl Experiment {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "growing" => Ok(Self::Growing),
            "persistent" => Ok(Self::Persistent),
            "regenerating" => Ok(Self::Regenerating),
            other => Err(format!("unknown experiment {other:?}")),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Growing => "growing",
            Self::Persistent => "persistent",
            Self::Regenerating => "regenerating",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TrainStats {
    pub final_loss: Option<f64>,
    pub steps: Option<u64>,
}

/// Premultiplied RGBA, `[y][x][rgba]`.
#[derive(Clone, Debug)]
pub struct Target {
    pub size: usize,
    pub rgba: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct Model {
    pub name: String,
    pub emoji_name: String,
    pub experiment: Experiment,
    pub fire_rate: f32,
    pub alive_threshold: f32,
    pub grid: usize,
    /// `w1` transposed to `[PERCEPTION][HIDDEN]` so the hidden-unit accumulate is contiguous.
    pub w1t: Box<[[f32; HIDDEN]; PERCEPTION]>,
    pub b1: [f32; HIDDEN],
    /// `w2` transposed to `[HIDDEN][CHANNELS]` so zero (relu'd) hidden units can be skipped.
    pub w2t: Box<[[f32; CHANNELS]; HIDDEN]>,
    pub target: Target,
    pub train: TrainStats,
}

#[derive(Deserialize)]
struct RawModel {
    format: String,
    name: String,
    emoji_name: String,
    experiment: String,
    channels: usize,
    hidden: usize,
    fire_rate: f32,
    alive_threshold: f32,
    grid: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    target_size: usize,
    target_rgba: Vec<f32>,
    #[serde(default)]
    train: serde_json::Value,
}

impl Model {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        Self::from_json(value)
    }

    pub fn from_json(value: serde_json::Value) -> Result<Self, String> {
        let raw: RawModel = serde_json::from_value(value).map_err(|e| e.to_string())?;
        if raw.format != "nca-v1" {
            return Err(format!("unsupported format {:?}", raw.format));
        }
        if raw.channels != CHANNELS || raw.hidden != HIDDEN {
            return Err(format!(
                "expected {CHANNELS} channels and {HIDDEN} hidden, got {} and {}",
                raw.channels, raw.hidden
            ));
        }
        expect_len("w1", &raw.w1, HIDDEN * PERCEPTION)?;
        expect_len("b1", &raw.b1, HIDDEN)?;
        expect_len("w2", &raw.w2, CHANNELS * HIDDEN)?;
        expect_len("target_rgba", &raw.target_rgba, raw.target_size * raw.target_size * 4)?;

        let mut w1t = Box::new([[0.0; HIDDEN]; PERCEPTION]);
        for (j, row) in raw.w1.chunks_exact(PERCEPTION).enumerate() {
            for (i, &v) in row.iter().enumerate() {
                w1t[i][j] = v;
            }
        }
        let mut w2t = Box::new([[0.0; CHANNELS]; HIDDEN]);
        for (c, row) in raw.w2.chunks_exact(HIDDEN).enumerate() {
            for (j, &v) in row.iter().enumerate() {
                w2t[j][c] = v;
            }
        }
        Ok(Self {
            name: raw.name,
            emoji_name: raw.emoji_name,
            experiment: Experiment::parse(&raw.experiment)?,
            fire_rate: raw.fire_rate,
            alive_threshold: raw.alive_threshold,
            grid: raw.grid,
            w1t,
            b1: raw.b1.try_into().expect("length checked"),
            w2t,
            target: Target { size: raw.target_size, rgba: raw.target_rgba },
            train: TrainStats {
                final_loss: raw.train.get("final_loss").and_then(|v| v.as_f64()),
                steps: raw.train.get("steps").and_then(|v| v.as_u64()),
            },
        })
    }
}

fn expect_len(field: &str, v: &[f32], n: usize) -> Result<(), String> {
    if v.len() == n {
        Ok(())
    } else {
        Err(format!("{field} has {} values, expected {n}", v.len()))
    }
}
