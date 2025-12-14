use std::sync::atomic::{AtomicBool, Ordering};

/// 🔥 Metrics yang dikirim dari training thread ke GUI
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
    pub is_running: AtomicBool,
    pub is_paused: AtomicBool,

    /// 🔴 Update setiap batch (realtime)
    pub current_metrics: Option<TrainingMetrics>,

    /// 🔵 Push hanya di akhir epoch (untuk plot)
    pub metrics_history: Vec<TrainingMetrics>,
}

impl TrainingState {
    pub fn new() -> Self {
        Self {
            is_running: AtomicBool::new(false),
            is_paused: AtomicBool::new(false),
            current_metrics: None,
            metrics_history: Vec::new(),
        }
    }

    /// ✅ Dipanggil SETIAP BATCH
    pub fn update_batch_metrics(&mut self, metrics: TrainingMetrics) {
        self.current_metrics = Some(metrics);
    }

    /// ✅ Dipanggil SETIAP AKHIR EPOCH
    pub fn push_epoch_metrics(&mut self, metrics: TrainingMetrics) {
        self.current_metrics = Some(metrics.clone());
        self.metrics_history.push(metrics);
        
        // 🔥 DEBUG: Confirm push
        log::info!(
            "📊 Epoch {} pushed. History len: {}",
            self.current_metrics.as_ref().unwrap().epoch,
            self.metrics_history.len()
        );
    }

    // ===== CONTROL =====
    pub fn start(&self) {
        self.is_running.store(true, Ordering::SeqCst);
        self.is_paused.store(false, Ordering::SeqCst);
    }

    pub fn stop(&self) {
        self.is_running.store(false, Ordering::SeqCst);
    }

    pub fn pause(&self) {
        self.is_paused.store(true, Ordering::SeqCst);
    }

    pub fn resume(&self) {
        self.is_paused.store(false, Ordering::SeqCst);
    }

    pub fn is_training(&self) -> bool {
        self.is_running.load(Ordering::SeqCst)
            && !self.is_paused.load(Ordering::SeqCst)
    }
}