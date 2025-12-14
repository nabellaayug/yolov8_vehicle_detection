use crate::model::YOLOv8Nano;
use crate::YOLOLoss;
use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::GradientsParams;
use burn::optim::Optimizer;
use burn::optim::{Adam, AdamConfig};
use burn::prelude::*;
use burn::tensor::backend::{AutodiffBackend, Backend};
pub mod state;

pub use state::{TrainingMetrics, TrainingState};

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub num_epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub num_classes: usize,
}

pub struct Trainer<B>
where
    B: AutodiffBackend,
{
    pub model: YOLOv8Nano<B>,
    pub optimizer: OptimizerAdaptor<Adam, YOLOv8Nano<B>, B>,
    pub config: TrainingConfig,
}

impl<B> Trainer<B>
where
    B: AutodiffBackend,
{
    pub fn new(device: &B::Device, config: TrainingConfig) -> Self {
        let model = YOLOv8Nano::new(device, config.num_classes);

        // 🔥 INI KUNCI
        let optimizer = AdamConfig::new().init::<B, YOLOv8Nano<B>>();

        Self {
            model,
            optimizer,
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
    let (p2, p3, p4) = self.model.forward(images);

    let loss = YOLOLoss::compute(
        p2,
        p3,
        p4,
        targets_p2,
        targets_p3,
        targets_p4,
    );

    let loss_value = loss.clone().into_scalar().to_f32();

     // 1️⃣ Backward → Gradients
    let grads = loss.backward();

    // 2️⃣ Convert → GradientsParams 
    let grads = GradientsParams::from_grads(grads, &self.model);

    // 3️⃣ Optimizer step (return model baru)
    self.model = self.optimizer.step(
        self.config.learning_rate as f64,
        self.model.clone(),
        grads,
    );

    loss_value
}

}
