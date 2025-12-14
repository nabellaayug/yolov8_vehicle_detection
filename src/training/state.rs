use serde::{Serialize, Deserialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub epoch: u32,
    pub total_epochs: u32,
    pub train_loss: f32,
    pub val_loss: f32,
    pub learning_rate: f32,
    pub batch_processed: u32,
    pub total_batches: u32,
}

#[derive(Debug)]
pub struct TrainingState {
    pub is_running: Arc<AtomicBool>,
    pub is_paused: Arc<AtomicBool>,
    pub current_epoch: Arc<AtomicU32>,
    pub metrics_history: Vec<TrainingMetrics>,
    pub current_metrics: Option<TrainingMetrics>,
    pub best_val_loss: f32,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            is_running: Arc::new(AtomicBool::new(false)),
            is_paused: Arc::new(AtomicBool::new(false)),
            current_epoch: Arc::new(AtomicU32::new(0)),
            metrics_history: Vec::new(),
            current_metrics: None,
            best_val_loss: f32::MAX,
        }
    }
}

impl TrainingState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn start(&self) {
        self.is_running.store(true, Ordering::SeqCst);
        self.is_paused.store(false, Ordering::SeqCst);
    }

    pub fn stop(&self) {
        self.is_running.store(false, Ordering::SeqCst);
        self.is_paused.store(false, Ordering::SeqCst);
    }

    pub fn pause(&self) {
        if self.is_running.load(Ordering::SeqCst) {
            self.is_paused.store(true, Ordering::SeqCst);
        }
    }

    pub fn resume(&self) {
        if self.is_running.load(Ordering::SeqCst) {
            self.is_paused.store(false, Ordering::SeqCst);
        }
    }

    pub fn is_training(&self) -> bool {
        self.is_running.load(Ordering::SeqCst) && !self.is_paused.load(Ordering::SeqCst)
    }

    pub fn update_metrics(&mut self, metrics: TrainingMetrics) {
        self.current_metrics = Some(metrics.clone());
        
        if metrics.val_loss < self.best_val_loss {
            self.best_val_loss = metrics.val_loss;
        }
        
        self.metrics_history.push(metrics);
    }

    pub fn get_progress(&self) -> f32 {
        if let Some(metrics) = &self.current_metrics {
            if metrics.total_epochs > 0 {
                return (metrics.epoch as f32 / metrics.total_epochs as f32).min(1.0);
            }
        }
        0.0
    }
}