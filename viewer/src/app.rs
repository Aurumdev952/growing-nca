use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use eframe::egui::{
    self, Color32, ColorImage, Key, Modifiers, Pos2, Rect, RichText, Sense, Stroke, TextureHandle, TextureOptions, pos2,
    vec2,
};

use crate::model::{CHANNELS, Model};
use crate::nca::{Grid, Sim};

const GRID_SIZES: [usize; 5] = [72, 96, 128, 192, 256];

pub fn run(models_dir: PathBuf) -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1180.0, 820.0]).with_title("Growing NCA"),
        ..Default::default()
    };
    eframe::run_native("Growing NCA", options, Box::new(|_| Ok(Box::new(App::new(models_dir)))))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ViewMode {
    Rgb,
    Alpha,
    HiddenRgb,
    Hidden(u8),
}

impl ViewMode {
    fn all() -> impl Iterator<Item = Self> {
        [Self::Rgb, Self::Alpha, Self::HiddenRgb].into_iter().chain((4..CHANNELS as u8).map(Self::Hidden))
    }

    fn label(self) -> String {
        match self {
            Self::Rgb => "RGB over white".into(),
            Self::Alpha => "Alpha".into(),
            Self::HiddenRgb => "Hidden 4,5,6 as RGB".into(),
            Self::Hidden(c) => format!("Hidden channel {c}"),
        }
    }
}

struct Controls {
    running: bool,
    steps_per_frame: u32,
    fire_rate: f32,
    rotation_deg: f32,
    grid_size: usize,
    brush_radius: f32,
    view: ViewMode,
}

struct Entry {
    path: PathBuf,
    model: Model,
    thumb: Option<TextureHandle>,
}

impl Entry {
    fn thumb(&mut self, ctx: &egui::Context) -> egui::TextureId {
        let model = &self.model;
        self.thumb.get_or_insert_with(|| ctx.load_texture(&model.name, target_image(model), TextureOptions::LINEAR)).id()
    }
}

#[derive(Default)]
struct Catalog {
    entries: Vec<Entry>,
    errors: Vec<(PathBuf, String)>,
}

impl Catalog {
    fn scan(dir: &Path) -> Self {
        let mut catalog = Self::default();
        let mut paths: Vec<PathBuf> = match std::fs::read_dir(dir) {
            Ok(rd) => rd.filter_map(|e| e.ok().map(|e| e.path())).collect(),
            Err(e) => {
                catalog.errors.push((dir.to_path_buf(), e.to_string()));
                return catalog;
            }
        };
        paths.retain(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            name.ends_with(".json") && !name.ends_with(".loss.json")
        });
        for path in paths {
            match Model::load(&path) {
                Ok(model) => catalog.entries.push(Entry { path, model, thumb: None }),
                Err(e) => catalog.errors.push((path, e)),
            }
        }
        catalog.entries.sort_by(|a, b| {
            (&a.model.emoji_name, a.model.experiment, &a.model.name).cmp(&(
                &b.model.emoji_name,
                b.model.experiment,
                &b.model.name,
            ))
        });
        catalog.errors.sort();
        catalog
    }
}

/// Wall-clock simulation throughput, refreshed every half second.
struct RateMeter {
    since: Instant,
    steps: u64,
    busy: Duration,
    steps_per_sec: f64,
    ms_per_step: f64,
}

impl RateMeter {
    fn new() -> Self {
        Self { since: Instant::now(), steps: 0, busy: Duration::ZERO, steps_per_sec: 0.0, ms_per_step: 0.0 }
    }

    fn record(&mut self, steps: u64, busy: Duration) {
        self.steps += steps;
        self.busy += busy;
        let elapsed = self.since.elapsed();
        if elapsed >= Duration::from_millis(500) {
            self.steps_per_sec = self.steps as f64 / elapsed.as_secs_f64();
            if self.steps > 0 {
                self.ms_per_step = self.busy.as_secs_f64() * 1e3 / self.steps as f64;
            }
            *self = Self { steps_per_sec: self.steps_per_sec, ms_per_step: self.ms_per_step, ..Self::new() };
        }
    }
}

struct App {
    models_dir: PathBuf,
    catalog: Catalog,
    selected: Option<usize>,
    sim: Sim,
    controls: Controls,
    rate: RateMeter,
    canvas: Option<TextureHandle>,
    last_brush: Option<(f32, f32)>,
}

impl App {
    fn new(models_dir: PathBuf) -> Self {
        let controls = Controls {
            running: true,
            steps_per_frame: 4,
            fire_rate: 0.5,
            rotation_deg: 0.0,
            grid_size: GRID_SIZES[0],
            brush_radius: 6.0,
            view: ViewMode::Rgb,
        };
        let mut app = Self {
            catalog: Catalog::default(),
            selected: None,
            sim: Sim::new(controls.grid_size),
            controls,
            rate: RateMeter::new(),
            canvas: None,
            last_brush: None,
            models_dir,
        };
        app.rescan();
        app
    }

    fn model(&self) -> Option<&Model> {
        self.selected.map(|i| &self.catalog.entries[i].model)
    }

    fn rescan(&mut self) {
        let current = self.selected.map(|i| self.catalog.entries[i].path.clone());
        self.catalog = Catalog::scan(&self.models_dir);
        let entries = &self.catalog.entries;
        match current.and_then(|p| entries.iter().position(|e| e.path == p)) {
            Some(i) => self.selected = Some(i),
            None => {
                let preferred = entries.iter().position(|e| e.model.name == "gecko_regenerating");
                self.select(preferred.or((!entries.is_empty()).then_some(0)));
            }
        }
    }

    fn select(&mut self, index: Option<usize>) {
        self.selected = index;
        if let Some(model) = self.model() {
            self.controls.fire_rate = model.fire_rate;
        }
        self.reset();
    }

    fn reset(&mut self) {
        self.sim = Sim::new(self.controls.grid_size);
        self.last_brush = None;
    }

    fn step_once(&mut self) {
        let Some(i) = self.selected else { return };
        let started = Instant::now();
        self.sim.advance(&self.catalog.entries[i].model, self.controls.rotation_deg.to_radians(), self.controls.fire_rate);
        self.rate.record(1, started.elapsed());
    }

    fn handle_keys(&mut self, ctx: &egui::Context) {
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        let (space, s, r) = ctx.input_mut(|i| {
            (
                i.consume_key(Modifiers::NONE, Key::Space),
                i.consume_key(Modifiers::NONE, Key::S),
                i.consume_key(Modifiers::NONE, Key::R),
            )
        });
        if space {
            self.controls.running = !self.controls.running;
        }
        if s {
            self.controls.running = false;
            self.step_once();
        }
        if r {
            self.reset();
        }
    }

    fn advance_frame(&mut self, ctx: &egui::Context) {
        if !self.controls.running || self.selected.is_none() {
            self.rate.record(0, Duration::ZERO);
            return;
        }
        for _ in 0..self.controls.steps_per_frame {
            self.step_once();
        }
        ctx.request_repaint();
    }

    fn models_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Models");
            if ui.button("Rescan").clicked() {
                self.rescan();
            }
        });
        ui.label(RichText::new(self.models_dir.display().to_string()).small().weak());
        if self.catalog.entries.is_empty() {
            ui.label("No models yet. Rescan once training writes one.");
        }
        let (cur_emoji, cur_exp) = self.model().map(|m| (m.emoji_name.clone(), m.experiment)).unzip();
        let mut clicked = None;
        ui.horizontal_wrapped(|ui| {
            let mut prev: Option<&str> = None;
            for (i, entry) in self.catalog.entries.iter_mut().enumerate() {
                if prev == Some(entry.model.emoji_name.as_str()) {
                    continue;
                }
                let icon = entry.thumb(ui.ctx());
                let on = cur_emoji.as_ref() == Some(&entry.model.emoji_name);
                let button = egui::Button::image((icon, vec2(28.0, 28.0))).selected(on);
                if ui.add(button).on_hover_text(&entry.model.emoji_name).clicked() {
                    clicked = Some(i);
                }
                prev = Some(entry.model.emoji_name.as_str());
            }
        });
        // Switching emoji keeps the current regime when the new emoji has a model for it.
        if let Some(first) = clicked {
            let name = &self.catalog.entries[first].model.emoji_name;
            clicked = self.catalog.entries.iter().position(|e| &e.model.emoji_name == name && Some(e.model.experiment) == cur_exp);
            clicked = clicked.or(Some(first));
        }
        if let Some(emoji) = &cur_emoji {
            ui.horizontal(|ui| {
                for (i, e) in self.catalog.entries.iter().enumerate().filter(|(_, e)| &e.model.emoji_name == emoji) {
                    if ui.selectable_label(self.selected == Some(i), e.model.experiment.label()).clicked() {
                        clicked = Some(i);
                    }
                }
            });
        }
        if let Some(i) = clicked.filter(|&i| Some(i) != self.selected) {
            self.select(Some(i));
        }
        if !self.catalog.errors.is_empty() {
            ui.collapsing(format!("{} unreadable files", self.catalog.errors.len()), |ui| {
                for (path, err) in &self.catalog.errors {
                    let name = path.file_name().map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into());
                    ui.label(RichText::new(format!("{name}: {err}")).small());
                }
            });
        }

        let Some(i) = self.selected else { return };
        let entry = &mut self.catalog.entries[i];
        let thumb = entry.thumb(ui.ctx());
        let model = &entry.model;
        ui.separator();
        ui.horizontal(|ui| {
            ui.image((thumb, vec2(96.0, 96.0)));
            ui.vertical(|ui| {
                ui.label(RichText::new(&model.name).strong());
                let stats = &model.train;
                ui.label(format!("train steps {}", stats.steps.map_or("?".into(), |s| s.to_string())));
                match stats.final_loss {
                    Some(l) => ui.label(format!("final loss {l:.5}\nlog10 {:.2}", l.log10())),
                    None => ui.label("final loss ?"),
                };
                ui.label(format!("trained on {0}x{0}", model.grid));
            });
        });
    }

    fn controls_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Simulation");
        ui.horizontal(|ui| {
            let label = if self.controls.running { "Pause (Space)" } else { "Play (Space)" };
            if ui.button(label).clicked() {
                self.controls.running = !self.controls.running;
            }
            if ui.button("Step (S)").clicked() {
                self.controls.running = false;
                self.step_once();
            }
            if ui.button("Reset (R)").clicked() {
                self.reset();
            }
        });
        let c = &mut self.controls;
        ui.add(egui::Slider::new(&mut c.steps_per_frame, 1..=64).text("steps / frame"));
        let default_rate = self.selected.map(|i| self.catalog.entries[i].model.fire_rate);
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut c.fire_rate, 0.0..=1.0).text("fire rate"));
            if let Some(rate) = default_rate.filter(|&r| r != c.fire_rate)
                && ui.small_button("model").clicked()
            {
                c.fire_rate = rate;
            }
        });
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut c.rotation_deg, -180.0..=180.0).text("rotation").suffix("°"));
            if c.rotation_deg != 0.0 && ui.small_button("0").clicked() {
                c.rotation_deg = 0.0;
            }
        });
        let mut size = c.grid_size;
        ui.horizontal(|ui| {
            ui.label("grid");
            for s in GRID_SIZES {
                ui.selectable_value(&mut size, s, s.to_string());
            }
        });
        if size != self.controls.grid_size {
            self.controls.grid_size = size;
            self.reset();
        }

        ui.add_space(6.0);
        ui.heading("Damage");
        ui.add(egui::Slider::new(&mut self.controls.brush_radius, 1.0..=30.0).text("brush radius"));
        ui.label(RichText::new("Left-drag erases. Right-click plants a seed.").small().weak());
        ui.horizontal(|ui| {
            if ui.button("Random damage").clicked() {
                self.sim.random_damage();
            }
            if ui.button("Cut left half").clicked() {
                let half = self.sim.grid.w / 2;
                self.sim.grid.clear_where(|_, x| x < half);
            }
        });

        ui.add_space(6.0);
        ui.heading("View");
        let view = &mut self.controls.view;
        egui::ComboBox::from_id_salt("view").selected_text(view.label()).show_ui(ui, |ui| {
            for mode in ViewMode::all() {
                ui.selectable_value(view, mode, mode.label());
            }
        });
    }

    fn stats_ui(&self, ui: &mut egui::Ui) {
        ui.heading("Stats");
        let Some(model) = self.model() else { return };
        let grid = &self.sim.grid;
        egui::Grid::new("stats").num_columns(2).show(ui, |ui| {
            ui.label("step");
            ui.label(self.sim.steps.to_string());
            ui.end_row();
            ui.label("steps / s");
            ui.label(if self.controls.running { format!("{:.0}", self.rate.steps_per_sec) } else { "paused".into() });
            ui.end_row();
            ui.label("cost / step");
            ui.label(format!("{:.3} ms", self.rate.ms_per_step));
            ui.end_row();
            ui.label("alive cells");
            ui.label(grid.alive_count(model.alive_threshold).to_string());
            ui.end_row();
            let mse = grid.mse(&model.target);
            ui.label("MSE vs target");
            ui.label(format!("{mse:.2e} (log10 {:.2})", mse.log10()));
            ui.end_row();
        });
        if self.controls.rotation_deg != 0.0 {
            ui.label(RichText::new("MSE assumes an unrotated pattern.").small().weak());
        }
    }

    fn canvas_ui(&mut self, ui: &mut egui::Ui) {
        let grid = &self.sim.grid;
        let image = render(grid, self.controls.view);
        let texture = self.canvas.get_or_insert_with(|| ui.ctx().load_texture("grid", image.clone(), TextureOptions::NEAREST));
        texture.set(image, TextureOptions::NEAREST);

        let (area, response) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let scale = (area.width() / grid.w as f32).min(area.height() / grid.h as f32);
        let rect = Rect::from_center_size(area.center(), vec2(grid.w as f32, grid.h as f32) * scale);
        let painter = ui.painter_at(area);
        painter.image(texture.id(), rect, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);

        let to_cell = |p: Pos2| ((p.y - rect.min.y) / scale, (p.x - rect.min.x) / scale);
        let radius = self.controls.brush_radius;
        if let Some(p) = response.hover_pos() {
            painter.circle_stroke(p, radius * scale, Stroke::new(1.5, Color32::from_rgb(220, 60, 60)));
        }

        let erasing = response.is_pointer_button_down_on() && ui.input(|i| i.pointer.primary_down());
        match response.interact_pointer_pos().filter(|_| erasing) {
            Some(p) => {
                let (y, x) = to_cell(p);
                let (y0, x0) = self.last_brush.unwrap_or((y, x));
                let dist = ((y - y0).powi(2) + (x - x0).powi(2)).sqrt();
                let n = (dist / (radius * 0.5).max(0.5)).ceil().max(1.0) as usize;
                for k in 0..=n {
                    let t = k as f32 / n as f32;
                    self.sim.grid.erase_disc(y0 + (y - y0) * t, x0 + (x - x0) * t, radius);
                }
                self.last_brush = Some((y, x));
            }
            None => self.last_brush = None,
        }

        if response.secondary_clicked()
            && let Some(p) = response.interact_pointer_pos()
        {
            let (y, x) = to_cell(p);
            let g = &mut self.sim.grid;
            if (0.0..g.h as f32).contains(&y) && (0.0..g.w as f32).contains(&x) {
                g.plant_seed(y as usize, x as usize);
            }
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.handle_keys(&ctx);
        self.advance_frame(&ctx);
        egui::Panel::left("side").exact_size(310.0).resizable(false).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.models_ui(ui);
                ui.separator();
                self.controls_ui(ui);
                ui.separator();
                self.stats_ui(ui);
            });
        });
        egui::CentralPanel::default().show(ui, |ui| {
            if self.selected.is_none() {
                ui.label(format!("No model loaded from {}.", self.models_dir.display()));
            }
            self.canvas_ui(ui);
        });
    }
}

fn gray(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

fn over_white(rgba: &[f32]) -> Color32 {
    let a = rgba[3].clamp(0.0, 1.0);
    Color32::from_rgb(gray(1.0 - a + rgba[0]), gray(1.0 - a + rgba[1]), gray(1.0 - a + rgba[2]))
}

fn diverging(t: f32) -> Color32 {
    let (end, t) = if t < 0.0 { ([0.23, 0.30, 0.75], -t) } else { ([0.71, 0.02, 0.15], t) };
    let mix = |e: f32| gray(1.0 + (e - 1.0) * t.min(1.0));
    Color32::from_rgb(mix(end[0]), mix(end[1]), mix(end[2]))
}

fn render(grid: &Grid, view: ViewMode) -> ColorImage {
    let cells = grid.state.chunks_exact(CHANNELS);
    let pixels: Vec<Color32> = match view {
        ViewMode::Rgb => cells.map(over_white).collect(),
        ViewMode::Alpha => cells.map(|c| Color32::from_gray(gray(c[3]))).collect(),
        ViewMode::HiddenRgb => {
            let f = |v: f32| gray(0.5 + 0.5 * v);
            cells.map(|c| Color32::from_rgb(f(c[4]), f(c[5]), f(c[6]))).collect()
        }
        ViewMode::Hidden(ch) => {
            let ch = ch as usize;
            let scale = cells.clone().map(|c| c[ch].abs()).fold(1e-6, f32::max);
            cells.map(|c| diverging(c[ch] / scale)).collect()
        }
    };
    ColorImage::new([grid.w, grid.h], pixels)
}

fn target_image(model: &Model) -> ColorImage {
    let n = model.target.size;
    ColorImage::new([n, n], model.target.rgba.chunks_exact(4).map(over_white).collect())
}
