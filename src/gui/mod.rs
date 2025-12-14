use crate::training::{TrainingState, TrainingMetrics};
use eframe::egui;
use egui_plot::{Corner, Legend, Line, Plot, PlotPoints};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

pub struct TrainingVisualizerApp {
    training_state: Arc<Mutex<TrainingState>>,
    // ✅ Cache data lokal jadi gak perlu lock terus
    cached_metrics: Vec<TrainingMetrics>,
    cached_current: Option<TrainingMetrics>,
}

impl TrainingVisualizerApp {
    pub fn new(training_state: Arc<Mutex<TrainingState>>) -> Self {
        Self {
            training_state,
            cached_metrics: Vec::new(),
            cached_current: None,
        }
    }
}

impl eframe::App for TrainingVisualizerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after(std::time::Duration::from_millis(100));

        // ✅ CEPAT: Ambil data sekali, langsung lepas lock
        let (current_metrics, metrics_history, is_running, is_paused) = {
            let state = self.training_state.lock().unwrap();
            (
                state.current_metrics.clone(),
                state.metrics_history.clone(),
                state.is_running.load(Ordering::SeqCst),
                state.is_paused.load(Ordering::SeqCst),
            )
        }; // ✅ LOCK LEPAS DI SINI

        // ✅ Update cache (gak perlu lock lagi)
        if let Some(ref m) = current_metrics {
            self.cached_current = Some(m.clone());
        }
        self.cached_metrics = metrics_history;

        // ===== UI RENDER (TANPA LOCK) =====
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("🔥 YOLOv8 Training Monitor");
            ui.separator();

            // ===== STATUS BAR =====
            {
                ui.horizontal(|ui| {
                    if let Some(m) = &self.cached_current {
                        ui.label(format!(
                            "Epoch {}/{} | Batch {}/{} | Loss {:.5}",
                            m.epoch, m.total_epochs, m.batch_processed, m.total_batches, m.train_loss
                        ));
                    } else {
                        ui.label("⏳ Waiting for training...");
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("⏹ Stop").clicked() {
                            self.training_state.lock().unwrap().stop();
                        }
                        if ui.button("⏸ Pause").clicked() {
                            self.training_state.lock().unwrap().pause();
                        }
                        if ui.button("▶ Resume").clicked() {
                            self.training_state.lock().unwrap().resume();
                        }
                        if ui.button("▶ Start").clicked() {
                            self.training_state.lock().unwrap().start();
                        }

                        if is_running && !is_paused {
                            ui.colored_label(egui::Color32::GREEN, "● Training");
                        } else if is_paused {
                            ui.colored_label(egui::Color32::YELLOW, "● Paused");
                        } else {
                            ui.colored_label(egui::Color32::RED, "● Stopped");
                        }
                    });
                });
            }

            ui.separator();

            // ===== LOSS PLOT =====
            let losses: Vec<f64> = self.cached_metrics
                .iter()
                .map(|m| m.train_loss as f64)
                .collect();

            if !losses.is_empty() {
                let points: PlotPoints = losses
                    .iter()
                    .enumerate()
                    .map(|(i, &l)| [i as f64, l])
                    .collect();

                Plot::new("loss_plot")
                    .legend(Legend::default().position(Corner::RightTop))
                    .height(300.0)
                    .show(ui, |plot_ui| {
                        plot_ui.line(Line::new(points).name("Train Loss"));
                    });

                // ✅ Tampilkan stats
                ui.horizontal(|ui| {
                    if let Some(last) = self.cached_metrics.last() {
                        ui.label(format!("Last epoch train loss: {:.5}", last.train_loss));
                        ui.label(format!("Last epoch val loss: {:.5}", last.val_loss));
                    }
                });
            } else {
                ui.label("⏳ Waiting for epoch completion...");
            }
        });
    }
}

pub fn run_gui(training_state: Arc<Mutex<TrainingState>>) -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions::default();

    eframe::run_native(
        "YOLOv8 Training Dashboard",
        options,
        Box::new(|_cc| Box::new(TrainingVisualizerApp::new(training_state))),
    )
}