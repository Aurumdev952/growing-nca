mod app;
mod model;
mod nca;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use model::Model;
use nca::Sim;

const USAGE: &str = "usage: nca-viewer [--models <dir>]
       nca-viewer --bench <model.json> [--grid N] [--steps N]";

enum Command {
    Gui { models: PathBuf },
    Bench { model: PathBuf, grid: usize, steps: u64 },
}

fn parse_args(mut args: impl Iterator<Item = String>) -> Result<Command, String> {
    let mut models = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../models"));
    let mut bench = None;
    let mut grid = 72;
    let mut steps = 2000;
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--models" => models = value()?.into(),
            "--bench" => bench = Some(PathBuf::from(value()?)),
            "--grid" => grid = value()?.parse().map_err(|e| format!("--grid: {e}"))?,
            "--steps" => steps = value()?.parse().map_err(|e| format!("--steps: {e}"))?,
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    if grid < 3 {
        return Err("--grid must be at least 3".into());
    }
    Ok(match bench {
        Some(model) => Command::Bench { model, grid, steps },
        None => Command::Gui { models: std::fs::canonicalize(&models).unwrap_or(models) },
    })
}

fn bench(path: &Path, grid: usize, steps: u64) -> Result<(), String> {
    let model = Model::load(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut sim = Sim::new(grid);
    println!("{} on {grid}x{grid}, {} threads", model.name, rayon::current_num_threads());
    for phase in ["from seed", "continued"] {
        let t = Instant::now();
        for _ in 0..steps {
            sim.advance(&model, 0.0, model.fire_rate);
        }
        let secs = t.elapsed().as_secs_f64();
        println!(
            "{phase:>9}: {steps} steps in {secs:.3}s = {:.0} steps/s, {} alive cells at the end",
            steps as f64 / secs,
            sim.grid.alive_count(model.alive_threshold)
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    let result = match parse_args(std::env::args().skip(1)) {
        Err(e) => Err(format!("{e}\n{USAGE}")),
        Ok(Command::Bench { model, grid, steps }) => bench(&model, grid, steps),
        Ok(Command::Gui { models }) => app::run(models).map_err(|e| e.to_string()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
