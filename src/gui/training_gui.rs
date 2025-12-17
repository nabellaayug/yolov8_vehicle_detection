use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};

pub struct TrainingGui {
    train_losses: Vec<f32>,
    val_losses: Vec<f32>,
    current_epoch: usize,
    total_epochs: usize,
    learning_rate: f64,
    status: String,
}

impl TrainingGui {
    pub fn new(total_epochs: usize, learning_rate: f64) -> Self {
        Self {
            train_losses: Vec::new(),
            val_losses: Vec::new(),
            current_epoch: 0,
            total_epochs,
            learning_rate,
            status: "Ready".to_string(),
        }
    }
    
    pub fn update_epoch(&mut self, epoch: usize, train_loss: f32, val_loss: f32) {
        self.current_epoch = epoch;
        self.train_losses.push(train_loss);
        self.val_losses.push(val_loss);
        self.status = format!("Training epoch {}/{}", epoch, self.total_epochs);
    }
    
    pub fn set_status(&mut self, status: String) {
        self.status = status;
    }
}

impl eframe::App for TrainingGui {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("🚀 YOLOv8 Training Monitor");
            ui.separator();
            
            // Status section
            ui.horizontal(|ui| {
                ui.label("Status:");
                ui.colored_label(egui::Color32::GREEN, &self.status);
            });
            
            ui.horizontal(|ui| {
                ui.label(format!("Epoch: {}/{}", self.current_epoch, self.total_epochs));
                ui.separator();
                ui.label(format!("Learning Rate: {:.6}", self.learning_rate));
            });
            
            if let (Some(&last_train), Some(&last_val)) = 
                (self.train_losses.last(), self.val_losses.last()) {
                ui.horizontal(|ui| {
                    ui.label(format!("Train Loss: {:.4}", last_train));
                    ui.separator();
                    ui.label(format!("Val Loss: {:.4}", last_val));
                });
                
                // Warning for overfitting
                if last_val > last_train * 1.2 {
                    ui.colored_label(
                        egui::Color32::RED,
                        "⚠️  Possible overfitting detected!"
                    );
                }
            }
            
            ui.separator();
            
            // Loss plot
            Plot::new("loss_plot")
                .height(400.0)
                .legend(egui_plot::Legend::default())
                .show(ui, |plot_ui| {
                    // Train loss line
                    let train_points: PlotPoints = self
                        .train_losses
                        .iter()
                        .enumerate()
                        .map(|(i, &loss)| [i as f64, loss as f64])
                        .collect();
                    
                    plot_ui.line(
                        Line::new(train_points)
                            .name("Train Loss")
                            .color(egui::Color32::BLUE)
                    );
                    
                    // Val loss line
                    let val_points: PlotPoints = self
                        .val_losses
                        .iter()
                        .enumerate()
                        .map(|(i, &loss)| [i as f64, loss as f64])
                        .collect();
                    
                    plot_ui.line(
                        Line::new(val_points)
                            .name("Val Loss")
                            .color(egui::Color32::RED)
                    );
                });
            
            ui.separator();
            
            // Progress bar
            let progress = self.current_epoch as f32 / self.total_epochs as f32;
            let progress_bar = egui::ProgressBar::new(progress)
                .text(format!("{:.1}%", progress * 100.0));
            ui.add(progress_bar);
        });
        
        // Request repaint untuk real-time updates
        ctx.request_repaint();
    }
}

pub fn launch_gui(total_epochs: usize, learning_rate: f64) -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_title("YOLOv8 Training"),
        ..Default::default()
    };
    
    eframe::run_native(
        "YOLOv8 Training Monitor",
        options,
        Box::new(move |_cc| {
            Ok(Box::new(TrainingGui::new(total_epochs, learning_rate)))
        }),
    )
}