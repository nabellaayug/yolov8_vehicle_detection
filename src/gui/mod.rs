use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use crate::training::{TrainingState, TrainingMetrics};
use std::sync::{Arc, Mutex};

pub struct TrainingVisualizerApp {
    training_state: Arc<Mutex<TrainingState>>,
    selected_metric: String,
}

impl Default for TrainingVisualizerApp {
    fn default() -> Self {
        Self {
            training_state: Arc::new(Mutex::new(TrainingState::new())),
            selected_metric: "train_loss".to_string(),
        }
    }
}

impl eframe::App for TrainingVisualizerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("🚀 YOLOv8 Training Monitor");
            
            self.render_controls(ui);
            ui.separator();
            self.render_metrics(ui);
            ui.separator();
            self.render_plot(ui);
        });

        // Auto refresh
        ctx.request_repaint();
    }
}

impl TrainingVisualizerApp {
    pub fn new(training_state: Arc<Mutex<TrainingState>>) -> Self {
        Self {
            training_state,
            selected_metric: "train_loss".to_string(),
        }
    }

    fn render_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let state = self.training_state.lock().unwrap();
            
            if ui.button("▶ Start Training").clicked() {
                state.start();
            }
            
            if ui.button("⏸ Pause").clicked() {
                state.pause();
            }
            
            if ui.button("▶ Resume").clicked() {
                state.resume();
            }
            
            if ui.button("⏹ Stop").clicked() {
                state.stop();
            }
            
            ui.separator();
            
            if state.is_training() {
                ui.colored_label(egui::Color32::GREEN, "● Training");
            } else if state.is_paused.load(std::sync::atomic::Ordering::SeqCst) {
                ui.colored_label(egui::Color32::YELLOW, "● Paused");
            } else {
                ui.colored_label(egui::Color32::RED, "● Stopped");
            }
        });
    }

    fn render_metrics(&mut self, ui: &mut egui::Ui) {
        let state = self.training_state.lock().unwrap();
        
        if let Some(metrics) = &state.current_metrics {
            ui.group(|ui| {
                ui.vertical(|ui| {
                    ui.label(format!("📊 Epoch: {}/{}", metrics.epoch, metrics.total_epochs));
                    
                    let progress = state.get_progress();
                    ui.add(
                        egui::ProgressBar::new(progress)
                            .show_percentage()
                            .desired_width(f32::INFINITY),
                    );
                    
                    ui.horizontal(|ui| {
                        ui.label("Batch Progress:");
                        ui.label(format!(
                            "{}/{}",
                            metrics.batch_processed, metrics.total_batches
                        ));
                    });
                });
            });
            
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.label("📈 Metrics:");
                        ui.label(format!("Train Loss: {:.6}", metrics.train_loss));
                        ui.label(format!("Val Loss: {:.6}", metrics.val_loss));
                        ui.label(format!("Best Val Loss: {:.6}", state.best_val_loss));
                    });
                    
                    ui.separator();
                    
                    ui.vertical(|ui| {
                        ui.label("⚙️ Config:");
                        ui.label(format!("Learning Rate: {:.6}", metrics.learning_rate));
                        ui.label(format!("Total Metrics: {}", state.metrics_history.len()));
                    });
                });
            });
        } else {
            ui.label("⏳ Waiting for training to start...");
        }
    }

    fn render_plot(&mut self, ui: &mut egui::Ui) {
        let state = self.training_state.lock().unwrap();
        
        ui.horizontal(|ui| {
            ui.label("Select Metric:");
            ui.selectable_value(&mut self.selected_metric, "train_loss".to_string(), "Train Loss");
            ui.selectable_value(&mut self.selected_metric, "val_loss".to_string(), "Val Loss");
        });

        let points = match self.selected_metric.as_str() {
            "train_loss" => {
                state.metrics_history
                    .iter()
                    .enumerate()
                    .map(|(i, m)| [i as f64, m.train_loss as f64])
                    .collect::<Vec<_>>()
            }
            "val_loss" => {
                state.metrics_history
                    .iter()
                    .enumerate()
                    .map(|(i, m)| [i as f64, m.val_loss as f64])
                    .collect::<Vec<_>>()
            }
            _ => vec![],
        };

        if !points.is_empty() {
            let line = Line::new(PlotPoints::new(points));
            Plot::new("training_plot")
                .legend(Default::default())
                .show(ui, |plot_ui| {
                    plot_ui.line(line);
                });
        } else {
            ui.vertical_centered(|ui| {
                ui.add_space(100.0);
                ui.heading("No data yet. Start training to see metrics!");
            });
        }
    }
}

pub fn run_gui(training_state: Arc<Mutex<TrainingState>>) -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0]),
        ..Default::default()
    };

    eframe::run_native(
        "YOLOv8 Training Monitor",
        options,
        Box::new(|_cc| Box::<TrainingVisualizerApp>::new(TrainingVisualizerApp::new(training_state))),
    )
}