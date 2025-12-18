use std::sync::{Arc, Mutex};
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};

/// Shared state antara trainer thread dan GUI thread
pub struct TrainingState {
    pub train_losses: Vec<f32>,
    pub val_losses: Vec<f32>,
    pub current_epoch: usize,
    pub total_epochs: usize,
    pub learning_rate: f64,
    pub status: String,
    pub is_training: bool,
}

pub struct TrainingGui {
    state: Arc<Mutex<TrainingState>>,
}

impl TrainingGui {
    pub fn new(total_epochs: usize, learning_rate: f64) -> Self {
        let state = TrainingState {
            train_losses: Vec::new(),
            val_losses: Vec::new(),
            current_epoch: 0,
            total_epochs,
            learning_rate,
            status: "Initializing...".to_string(),
            is_training: true,
        };
        
        Self {
            state: Arc::new(Mutex::new(state)),
        }
    }
    
    /// Create GUI from existing state (for main thread usage)
    pub fn from_state(state: Arc<Mutex<TrainingState>>) -> Self {
        Self { state }
    }
    
    pub fn get_state_clone(&self) -> Arc<Mutex<TrainingState>> {
        Arc::clone(&self.state)
    }
}

impl eframe::App for TrainingGui {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let state = self.state.lock().unwrap();
        
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("YOLOv8 Training Monitor");
            ui.separator();
            
            // Status section
            ui.horizontal(|ui| {
                ui.label("Status:");
                let color = if state.is_training {
                    egui::Color32::YELLOW
                } else {
                    egui::Color32::GREEN
                };
                ui.colored_label(color, &state.status);
            });
            
            ui.horizontal(|ui| {
                ui.label(format!("Epoch: {}/{}", state.current_epoch, state.total_epochs));
                ui.separator();
                ui.label(format!("Learning Rate: {:.6}", state.learning_rate));
            });
            
            if let (Some(&last_train), Some(&last_val)) = 
                (state.train_losses.last(), state.val_losses.last()) {
                ui.horizontal(|ui| {
                    //Handle train loss display
                    if last_train.is_finite() {
                        ui.label(format!("Train Loss: {:.4}", last_train));
                    } else if last_train.is_infinite() {
                        ui.colored_label(egui::Color32::YELLOW, "Train Loss: inf");
                    } else {
                        ui.colored_label(egui::Color32::RED, "Train Loss: NaN");
                    }
                    
                    ui.separator();
                    
                    //Handle val loss display
                    if last_val.is_finite() {
                        ui.label(format!("Val Loss: {:.4}", last_val));
                    } else if last_val.is_infinite() {
                        ui.colored_label(egui::Color32::YELLOW, "Val Loss: inf (no valid batches)");
                    } else {
                        ui.colored_label(egui::Color32::RED, "Val Loss: NaN");
                    }
                });
                
                //Only show warnings if both losses are finite
                if last_train.is_finite() && last_val.is_finite() {
                    // Warning for overfitting
                    if last_val > last_train * 1.2 {
                        ui.colored_label(
                            egui::Color32::RED,
                            "Possible overfitting detected (Val Loss > 1.2x Train Loss)!"
                        );
                    }
                    
                    // Improvement indicator
                    if state.train_losses.len() > 1 {
                        let prev_train = state.train_losses[state.train_losses.len() - 2];
                        if prev_train.is_finite() && last_train < prev_train {
                            ui.colored_label(
                                egui::Color32::GREEN,
                                format!("Train loss improved: {:.4} → {:.4}", prev_train, last_train)
                            );
                        } else if prev_train.is_finite() && last_train > prev_train {
                            ui.colored_label(
                                egui::Color32::from_rgb(255, 165, 0),  // Orange
                                format!("Train loss increased: {:.4} → {:.4}", prev_train, last_train)
                            );
                        }
                    }
                }
            }
            
            ui.separator();
            
            //Loss plot with inf/nan filtering
            Plot::new("loss_plot")
                .height(400.0)
                .show(ui, |plot_ui| {
                    // Train loss line
                    if !state.train_losses.is_empty() {
                        //Filter out NaN/Inf values to prevent crash
                        let train_data: Vec<[f64; 2]> = state
                            .train_losses
                            .iter()
                            .enumerate()
                            .filter(|(_, &loss)| loss.is_finite())  // KEY FIX
                            .map(|(i, &loss)| [i as f64 + 1.0, loss as f64])
                            .collect();
                        
                        if !train_data.is_empty() {
                            let train_points: PlotPoints = train_data.into();
                            plot_ui.line(
                                Line::new(train_points)
                                    .name("Train Loss")
                                    .color(egui::Color32::BLUE)
                                    .width(2.0)
                            );
                        }
                    }
                    
                    // Val loss line
                    if !state.val_losses.is_empty() {
                        //Filter out NaN/Inf values to prevent crash
                        let val_data: Vec<[f64; 2]> = state
                            .val_losses
                            .iter()
                            .enumerate()
                            .filter(|(_, &loss)| loss.is_finite())  // KEY FIX
                            .map(|(i, &loss)| [i as f64 + 1.0, loss as f64])
                            .collect();
                        
                        if !val_data.is_empty() {
                            let val_points: PlotPoints = val_data.into();
                            plot_ui.line(
                                Line::new(val_points)
                                    .name("Val Loss")
                                    .color(egui::Color32::RED)
                                    .width(2.0)
                            );
                        }
                    }
                });
            
            ui.separator();
            
            // Progress bar
            if state.total_epochs > 0 {
                let progress = state.current_epoch as f32 / state.total_epochs as f32;
                let progress_bar = egui::ProgressBar::new(progress)
                    .text(format!("{:.1}%", progress * 100.0));
                ui.add(progress_bar);
            }
            
            // Statistics with inf/nan handling
            ui.separator();
            ui.group(|ui| {
                ui.label("Statistics:");
                
                if !state.train_losses.is_empty() {
                    // Filter finite values for average
                    let finite_train: Vec<f32> = state.train_losses.iter()
                        .filter(|&&loss| loss.is_finite())
                        .copied()
                        .collect();
                    
                    if !finite_train.is_empty() {
                        let avg_train: f32 = finite_train.iter().sum::<f32>() / finite_train.len() as f32;
                        ui.label(format!("  Average Train Loss: {:.4}", avg_train));
                    } else {
                        ui.colored_label(
                            egui::Color32::YELLOW, 
                            "  Average Train Loss: N/A (no valid values)"
                        );
                    }
                }
                
                if !state.val_losses.is_empty() {
                    // Filter finite values for average
                    let finite_val: Vec<f32> = state.val_losses.iter()
                        .filter(|&&loss| loss.is_finite())
                        .copied()
                        .collect();
                    
                    if !finite_val.is_empty() {
                        let avg_val: f32 = finite_val.iter().sum::<f32>() / finite_val.len() as f32;
                        ui.label(format!("  Average Val Loss: {:.4}", avg_val));
                    } else {
                        ui.colored_label(
                            egui::Color32::YELLOW, 
                            "  Average Val Loss: N/A (no valid values)"
                        );
                    }
                }
                
                ui.label(format!("  Total Epochs: {}/{}", state.current_epoch, state.total_epochs));
            });
        });
        
        // Request repaint untuk real-time updates
        if state.is_training {
            ctx.request_repaint();
        }
    }
}

pub fn launch_gui(total_epochs: usize, learning_rate: f64) -> Arc<Mutex<TrainingState>> {
    eprintln!("WARNING: launch_gui() is deprecated on Windows!");
    eprintln!("   Use TrainingGui::from_state() in main thread instead.");
    eprintln!("   See train.rs for proper implementation.");
    
    let gui = TrainingGui::new(total_epochs, learning_rate);
    gui.get_state_clone()
}
