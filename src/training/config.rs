use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    // Dataset
    pub data_yaml: String,
    pub img_size: usize,
    pub num_classes: usize,
    
    // Training
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub warmup_epochs: usize,
    
    // Model
    pub reg_max: usize,
    
    // Loss weights
    pub box_loss_weight: f32,
    pub cls_loss_weight: f32,
    pub dfl_loss_weight: f32,
    
    // Inference
    pub conf_threshold: f32,
    pub iou_threshold: f32,
    
    // Early stopping
    pub patience: usize,
    pub min_delta: f32,
    
    // Checkpointing
    pub save_dir: String,
    pub save_interval: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            data_yaml: "data/Cars Detection/data.yaml".to_string(),
            img_size: 640,
            num_classes: 5,
            epochs: 10,
            batch_size: 1,
            learning_rate: 0.001,
            weight_decay: 0.0005,
            warmup_epochs: 3,
            reg_max: 16,
            box_loss_weight: 7.5,
            cls_loss_weight: 0.5,
            dfl_loss_weight: 1.5,
            conf_threshold: 0.25,
            iou_threshold: 0.45,
            patience: 10,
            min_delta: 0.001,
            save_dir: "runs/train".to_string(),
            save_interval: 10,
        }
    }
}

impl TrainingConfig {
    pub fn from_yaml(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: TrainingConfig = serde_yaml::from_str(&content)?;
        Ok(config)
    }
    
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let yaml = serde_yaml::to_string(self)?;
        std::fs::write(path, yaml)?;
        Ok(())
    }
}