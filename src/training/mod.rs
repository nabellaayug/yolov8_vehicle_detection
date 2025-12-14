pub mod state;

use burn::prelude::*;
use burn::optim::Adam;
use crate::model::{YOLOv8Nano, YOLOLoss};
pub use state::{TrainingState, TrainingMetrics};

pub struct TrainingConfig {
    pub num_epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub img_size: usize,
    pub num_classes: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            num_epochs: 100,
            batch_size: 16,
            learning_rate: 0.001,
            img_size: 640,
            num_classes: 80,
        }
    }
}

pub struct Trainer<B: Backend> {
    pub model: YOLOv8Nano<B>,
    pub config: TrainingConfig,
}

impl<B: Backend> Trainer<B> {
    pub fn new(device: &B::Device, config: TrainingConfig) -> Self {
        let model = YOLOv8Nano::new(device, config.num_classes);
        Self {
            model,
            config,
        }
    }

    pub fn train_step(
        &mut self,
        images: Tensor<B, 4>,
        targets_p2: Tensor<B, 4>,
        targets_p3: Tensor<B, 4>,
        targets_p4: Tensor<B, 4>,
    ) -> f32 {
        // Forward pass
        let (pred_p2, pred_p3, pred_p4) = self.model.forward(images);
        
        // Compute loss
        let loss = YOLOLoss::compute(
            pred_p2,
            pred_p3,
            pred_p4,
            targets_p2,
            targets_p3,
            targets_p4,
        );
        
        // Get loss value
        let loss_value = loss.into_scalar().to_f32();
        
        loss_value
    }

    pub fn train_epoch(&mut self, dataset: &[(Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>)]) -> f32 {
        let mut total_loss = 0.0;
        let mut count = 0;
        
        for (images, targets_p2, targets_p3, targets_p4) in dataset {
            let loss = self.train_step(
                images.clone(),
                targets_p2.clone(),
                targets_p3.clone(),
                targets_p4.clone(),
            );
            total_loss += loss;
            count += 1;
        }
        
        total_loss / count as f32
    }

    pub fn validate(&self, dataset: &[(Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>)]) -> f32 {
        let mut total_loss = 0.0;
        let mut count = 0;
        
        for (images, targets_p2, targets_p3, targets_p4) in dataset {
            let (pred_p2, pred_p3, pred_p4) = self.model.forward(images.clone());
            let loss = YOLOLoss::compute(
                pred_p2,
                pred_p3,
                pred_p4,
                targets_p2.clone(),
                targets_p3.clone(),
                targets_p4.clone(),
            );
            
            total_loss += loss.into_scalar().to_f32();
            count += 1;
        }
        
        total_loss / count as f32
    }
}