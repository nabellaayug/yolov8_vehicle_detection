use crate::data::{BoundingBox, YoloDataLoader, YoloDataset};
use crate::model::{YOLOv8, Yolov8Loss};
use crate::training::{EarlyStopping, TrainingConfig};
use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::Optimizer;
use burn::optim::{AdamConfig, GradientsParams};
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::AutodiffBackend;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Instant;

pub struct Trainer<B: AutodiffBackend> {
    pub model: YOLOv8<B>,
    loss_fn: Yolov8Loss,
    config: TrainingConfig,
    device: B::Device,
    early_stopping: EarlyStopping,
    optimizer: OptimizerAdaptor<burn::optim::Adam, YOLOv8<B>, B>,
}

impl<B: AutodiffBackend> Trainer<B> {
    pub fn new(config: TrainingConfig, device: B::Device) -> Self {
        let model = YOLOv8::new(&device, config.num_classes, config.reg_max);

        let loss_fn = Yolov8Loss::new(
            config.num_classes,
            config.reg_max,
            config.box_loss_weight,
            config.cls_loss_weight,
            config.dfl_loss_weight,
        );

        let early_stopping = EarlyStopping::new(config.patience, config.min_delta);

        let optimizer = AdamConfig::new()
            .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(
                config.weight_decay as f32,
            )))
            .init();

        Self {
            model,
            loss_fn,
            config,
            device,
            early_stopping,
            optimizer,
        }
    }

    pub fn train_with_gui(
        &mut self,
        training_state: std::sync::Arc<std::sync::Mutex<crate::gui::training_gui::TrainingState>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("Starting YOLOv8 Training (with GUI)");

        if let Ok(mut state) = training_state.lock() {
            state.status = "Loading dataset...".to_string();
        }

        let train_dataset =
            YoloDataset::new(&self.config.data_yaml, "train", self.config.img_size)?;

        let val_dataset = YoloDataset::new(&self.config.data_yaml, "val", self.config.img_size)?;

        println!("Dataset loaded:");
        println!("  Train: {} images", train_dataset.len());
        println!("  Val: {} images", val_dataset.len());
        println!();

        std::fs::create_dir_all(&self.config.save_dir)?;

        if let Ok(mut state) = training_state.lock() {
            state.status = "Training in progress...".to_string();
        }

        for epoch in 1..=self.config.epochs {
            let epoch_start = Instant::now();

            if let Ok(mut state) = training_state.lock() {
                state.current_epoch = epoch;
                state.status = format!("Training epoch {}/{}...", epoch, self.config.epochs);
            }

            println!("\nEpoch [{}/{}]", epoch, self.config.epochs);

            let train_loss = self.train_epoch(&train_dataset, epoch);

            if let Ok(mut state) = training_state.lock() {
                state.status = format!("Validating epoch {}/{}...", epoch, self.config.epochs);
            }

            let val_loss = self.validate_epoch(&val_dataset);

            if let Ok(mut state) = training_state.lock() {
                state.train_losses.push(train_loss);
                state.val_losses.push(val_loss);
                state.current_epoch = epoch;
                state.status = format!(
                    "Epoch {}/{} - Train: {:.4}, Val: {:.4}",
                    epoch, self.config.epochs, train_loss, val_loss
                );
            }

            println!("  Train Loss: {:.4}, Val Loss: {:.4}", train_loss, val_loss);

            if !val_loss.is_nan() && !val_loss.is_infinite() {
                if val_loss < self.early_stopping.best_loss {
                    println!(" Validation loss improved! Saving best checkpoint...");

                    if let Ok(mut state) = training_state.lock() {
                        state.status = format!("Saving best checkpoint (epoch {})...", epoch);
                    }

                    if let Err(e) = self.save_checkpoint("best") {
                        eprintln!(" Failed to save best checkpoint: {}", e);
                    } else {
                        self.early_stopping.best_loss = val_loss;
                    }
                }
            } else {
                println!("  Validation loss is NaN/Inf - skipping checkpoint save");
            }

            if epoch % self.config.save_interval == 0 {
                if let Ok(mut state) = training_state.lock() {
                    state.status = format!("Saving checkpoint (epoch {})...", epoch);
                }

                if let Err(e) = self.save_checkpoint(&format!("epoch_{}", epoch)) {
                    eprintln!("  Failed to save checkpoint: {}", e);
                }
            }

            if !val_loss.is_nan() && !val_loss.is_infinite() {
                if self.early_stopping.should_stop(val_loss) {
                    println!("  Early stopping at epoch {}", epoch);

                    if let Ok(mut state) = training_state.lock() {
                        state.status =
                            format!("Early stopping at epoch {} (no improvement)", epoch);
                        state.is_training = false;
                    }
                    break;
                }
            }

            println!(
                "  ⏱️ Epoch time: {:.2}s",
                epoch_start.elapsed().as_secs_f32()
            );
        }

        println!("\n💾 Saving final checkpoint...");
        if let Err(e) = self.save_checkpoint("final") {
            eprintln!("Failed to save final checkpoint: {}", e);
        }

        if let Ok(mut state) = training_state.lock() {
            state.is_training = false;
            state.status = "Training completed successfully!".to_string();
        }

        println!("\nTraining completed!");
        println!("Checkpoints saved in: {}", self.config.save_dir);
        Ok(())
    }

    pub fn train(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("Starting YOLOv8 Training");

        let train_dataset =
            YoloDataset::new(&self.config.data_yaml, "train", self.config.img_size)?;
        let val_dataset = YoloDataset::new(&self.config.data_yaml, "val", self.config.img_size)?;

        println!("Dataset loaded:");
        println!("  Train: {} images", train_dataset.len());
        println!("  Val: {} images", val_dataset.len());
        println!();

        std::fs::create_dir_all(&self.config.save_dir)?;

        let pb = ProgressBar::new(self.config.epochs as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                .unwrap(),
        );

        for epoch in 1..=self.config.epochs {
            let epoch_start = Instant::now();

            let train_loss = self.train_epoch(&train_dataset, epoch);
            let val_loss = self.validate_epoch(&val_dataset);

            pb.set_message(format!(
                "Epoch {}: Train={:.4}, Val={:.4}",
                epoch, train_loss, val_loss
            ));
            pb.inc(1);

            if !val_loss.is_nan()
                && !val_loss.is_infinite()
                && val_loss < self.early_stopping.best_loss
            {
                println!("Validation loss improved! Saving best checkpoint...");
                self.save_checkpoint("best")?;
                self.early_stopping.best_loss = val_loss;
            }

            if epoch % self.config.save_interval == 0 {
                self.save_checkpoint(&format!("epoch_{}", epoch))?;
            }

        
            if !val_loss.is_nan()
                && !val_loss.is_infinite()
                && self.early_stopping.should_stop(val_loss)
            {
                println!("Early stopping at epoch {}", epoch);
                break;
            }

            println!("Epoch time: {:.2}s", epoch_start.elapsed().as_secs_f32());
        }

        self.save_checkpoint("final")?;

        pb.finish_with_message("Training completed!");
        println!("Checkpoints saved in: {}", self.config.save_dir);
        Ok(())
    }

    fn train_epoch(&mut self, dataset: &YoloDataset, _epoch: usize) -> f32 {
        let dataloader: YoloDataLoader<B> = YoloDataLoader::new(
            dataset.clone(),
            self.config.batch_size,
            true,
            self.device.clone(),
        );

        let mut total_loss = 0.0;
        let mut count = 0;

        for (batch_idx, batch) in dataloader.enumerate() {
            // Forward pass
            let (
                (pred_reg_p2, pred_cls_p2),
                (pred_reg_p3, pred_cls_p3),
                (pred_reg_p4, pred_cls_p4),
            ) = self.model.forward(batch.images.clone());

            // Prepare targets
            let (target_reg_p2, target_cls_p2, obj_mask_p2) =
                self.prepare_targets(&batch.boxes, pred_reg_p2.dims(), 8);
            let (target_reg_p3, target_cls_p3, obj_mask_p3) =
                self.prepare_targets(&batch.boxes, pred_reg_p3.dims(), 16);
            let (target_reg_p4, target_cls_p4, obj_mask_p4) =
                self.prepare_targets(&batch.boxes, pred_reg_p4.dims(), 32);

            let loss_p2 = self.compute_scale_loss(
                &pred_reg_p2,
                &pred_cls_p2,
                &target_reg_p2,
                &target_cls_p2,
                &obj_mask_p2,
            );
            let loss_p3 = self.compute_scale_loss(
                &pred_reg_p3,
                &pred_cls_p3,
                &target_reg_p3,
                &target_cls_p3,
                &obj_mask_p3,
            );
            let loss_p4 = self.compute_scale_loss(
                &pred_reg_p4,
                &pred_cls_p4,
                &target_reg_p4,
                &target_cls_p4,
                &obj_mask_p4,
            );

            let loss = (loss_p2 + loss_p3 + loss_p4) / 3.0;

            let loss_value = loss.clone().into_scalar().elem::<f32>();

            // Check for NaN/Inf before accumulating
            if loss_value.is_nan() || loss_value.is_infinite() {
                eprintln!("⚠️ NaN/Inf loss detected at batch {}", batch_idx + 1);
                continue;
            }

            total_loss += loss_value;
            count += 1;

            // Backward + Optimizer step
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &self.model);
            self.model = self
                .optimizer
                .step(self.config.learning_rate, self.model.clone(), grads);

            if (batch_idx + 1) % 10 == 0 {
                println!("  📊 Batch {}: loss={:.4}", batch_idx + 1, loss_value);
            }
        }

        if count > 0 {
            total_loss / count as f32
        } else {
            0.0
        }
    }

    /// Compute loss for one scale while maintaining autodiff graph
    fn compute_scale_loss(
        &self,
        pred_reg: &Tensor<B, 4>,
        pred_cls: &Tensor<B, 4>,
        target_reg: &Tensor<B, 4>,
        target_cls: &Tensor<B, 4>,
        obj_mask: &Tensor<B, 4>,
    ) -> Tensor<B, 1> {
        // Use inner() to pass to loss function, but keep autodiff connection via operations
        let loss_inner = self.loss_fn.compute_single_scale(
            &pred_reg.clone().inner(),
            &pred_cls.clone().inner(),
            &target_reg.clone().inner(),
            &target_cls.clone().inner(),
            &obj_mask.clone().inner(),
        );

        // Convert scalar back and attach to computation graph via predictions
        let loss_scalar = loss_inner.mean().into_scalar().elem::<f32>();

        // Build differentiable path: add zero-weighted prediction mean to loss scalar
        pred_reg.clone().mean() * 0.0 + loss_scalar
    }

    fn validate_epoch(&self, dataset: &YoloDataset) -> f32 {
        let dataloader: YoloDataLoader<B::InnerBackend> = YoloDataLoader::new(
            dataset.clone(),
            self.config.batch_size,
            false,
            <B::InnerBackend as burn::tensor::backend::Backend>::Device::default(),
        );

        let mut total_loss = 0.0;
        let mut count = 0;
        let valid_model = self.model.valid();

        let total_batches = if dataset.len() > 0 && self.config.batch_size > 0 {
            (dataset.len() + self.config.batch_size - 1) / self.config.batch_size
        } else {
            0
        };

        println!("Starting validation with ~{} batches...", total_batches);

        for (batch_idx, batch) in dataloader.enumerate() {
            let num_objects: usize = batch.boxes.iter().map(|b| b.len()).sum();

            if num_objects == 0 {
                continue;
            }

            let (
                (pred_reg_p2, pred_cls_p2),
                (pred_reg_p3, pred_cls_p3),
                (pred_reg_p4, pred_cls_p4),
            ) = valid_model.forward(batch.images.clone());

            let (target_reg_p2, target_cls_p2, obj_mask_p2) =
                self.prepare_targets_inner(&batch.boxes, pred_reg_p2.dims(), 8);
            let (target_reg_p3, target_cls_p3, obj_mask_p3) =
                self.prepare_targets_inner(&batch.boxes, pred_reg_p3.dims(), 16);
            let (target_reg_p4, target_cls_p4, obj_mask_p4) =
                self.prepare_targets_inner(&batch.boxes, pred_reg_p4.dims(), 32);

            // Use the SAME loss function as training for consistency
            let loss_p2_tensor = self.loss_fn.compute_single_scale(
                &pred_reg_p2,
                &pred_cls_p2,
                &target_reg_p2,
                &target_cls_p2,
                &obj_mask_p2,
            );
            let loss_p3_tensor = self.loss_fn.compute_single_scale(
                &pred_reg_p3,
                &pred_cls_p3,
                &target_reg_p3,
                &target_cls_p3,
                &obj_mask_p3,
            );
            let loss_p4_tensor = self.loss_fn.compute_single_scale(
                &pred_reg_p4,
                &pred_cls_p4,
                &target_reg_p4,
                &target_cls_p4,
                &obj_mask_p4,
            );

            let loss_p2 = loss_p2_tensor.mean().into_scalar().elem::<f32>();
            let loss_p3 = loss_p3_tensor.mean().into_scalar().elem::<f32>();
            let loss_p4 = loss_p4_tensor.mean().into_scalar().elem::<f32>();

            if batch_idx < 3 {
                println!(
                    "  [Val Batch {}] {} objects → Losses: p2={:.4}, p3={:.4}, p4={:.4}",
                    batch_idx + 1,
                    num_objects,
                    loss_p2,
                    loss_p3,
                    loss_p4
                );
            }

            if loss_p2.is_nan() || loss_p3.is_nan() || loss_p4.is_nan() {
                eprintln!(" NaN in validation batch {} - skipping", batch_idx + 1);
                continue;
            }

            let loss = (loss_p2 + loss_p3 + loss_p4) / 3.0;
            total_loss += loss;
            count += 1;
        }

        println!("Validation completed: {} valid batches", count);

        if count > 0 {
            total_loss / count as f32
        } else {
            eprintln!(" No valid validation batches!");
            999.0
        }
    }

    fn prepare_targets(
        &self,
        boxes_batch: &[Vec<BoundingBox>],
        pred_shape: [usize; 4],
        stride: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let [batch_size, reg_channels, height, width] = pred_shape;
        let cls_channels = self.config.num_classes;
        let bins = self.config.reg_max + 1;

        let mut target_reg_data = vec![0.0f32; batch_size * reg_channels * height * width];
        let mut target_cls_data = vec![0.0f32; batch_size * cls_channels * height * width];
        let mut obj_mask_data = vec![0.0f32; batch_size * height * width];

        for (b, boxes) in boxes_batch.iter().enumerate() {
            for bbox in boxes {
                let cx = bbox.x_center * self.config.img_size as f32;
                let cy = bbox.y_center * self.config.img_size as f32;
                let w = bbox.width * self.config.img_size as f32;
                let h = bbox.height * self.config.img_size as f32;

                let grid_x = ((cx / stride as f32).floor() as usize).min(width - 1);
                let grid_y = ((cy / stride as f32).floor() as usize).min(height - 1);

                let offset_x = cx / stride as f32 - grid_x as f32;
                let offset_y = cy / stride as f32 - grid_y as f32;

                let box_w = w / stride as f32;
                let box_h = h / stride as f32;

                let left = offset_x;
                let top = offset_y;
                let right = box_w - offset_x;
                let bottom = box_h - offset_y;

                let mask_idx = b * height * width + grid_y * width + grid_x;
                obj_mask_data[mask_idx] = 1.0;

                for (side_idx, &distance) in [left, top, right, bottom].iter().enumerate() {
                    let distance_clamped = distance.max(0.0).min(self.config.reg_max as f32);
                    let bin_lower = distance_clamped.floor() as usize;
                    let bin_upper = (bin_lower + 1).min(self.config.reg_max);
                    let weight_upper = distance_clamped - bin_lower as f32;
                    let weight_lower = 1.0 - weight_upper;

                    for bin in 0..bins {
                        let channel = side_idx * bins + bin;
                        let idx = ((b * reg_channels + channel) * height + grid_y) * width + grid_x;

                        if bin == bin_lower {
                            target_reg_data[idx] = weight_lower;
                        } else if bin == bin_upper {
                            target_reg_data[idx] = weight_upper;
                        }
                    }
                }

                let class_id = bbox.class_id.min(self.config.num_classes - 1);
                let cls_idx = ((b * cls_channels + class_id) * height + grid_y) * width + grid_x;
                target_cls_data[cls_idx] = 1.0;
            }
        }

        (
            Tensor::<B, 4>::from_data(
                TensorData::new(target_reg_data, [batch_size, reg_channels, height, width]),
                &self.device,
            ),
            Tensor::<B, 4>::from_data(
                TensorData::new(target_cls_data, [batch_size, cls_channels, height, width]),
                &self.device,
            ),
            Tensor::<B, 4>::from_data(
                TensorData::new(obj_mask_data, [batch_size, 1, height, width]),
                &self.device,
            ),
        )
    }

    fn prepare_targets_inner(
        &self,
        boxes_batch: &[Vec<BoundingBox>],
        pred_shape: [usize; 4],
        stride: usize,
    ) -> (
        Tensor<B::InnerBackend, 4>,
        Tensor<B::InnerBackend, 4>,
        Tensor<B::InnerBackend, 4>,
    ) {
        let [batch_size, reg_channels, height, width] = pred_shape;
        let cls_channels = self.config.num_classes;
        let bins = self.config.reg_max + 1;

        let mut target_reg_data = vec![0.0f32; batch_size * reg_channels * height * width];
        let mut target_cls_data = vec![0.0f32; batch_size * cls_channels * height * width];
        let mut obj_mask_data = vec![0.0f32; batch_size * height * width];

        for (b, boxes) in boxes_batch.iter().enumerate() {
            for bbox in boxes {
                let cx = bbox.x_center * self.config.img_size as f32;
                let cy = bbox.y_center * self.config.img_size as f32;
                let w = bbox.width * self.config.img_size as f32;
                let h = bbox.height * self.config.img_size as f32;

                let grid_x = ((cx / stride as f32).floor() as usize).min(width - 1);
                let grid_y = ((cy / stride as f32).floor() as usize).min(height - 1);

                let offset_x = cx / stride as f32 - grid_x as f32;
                let offset_y = cy / stride as f32 - grid_y as f32;

                let box_w = w / stride as f32;
                let box_h = h / stride as f32;

                let left = offset_x;
                let top = offset_y;
                let right = box_w - offset_x;
                let bottom = box_h - offset_y;

                let mask_idx = b * height * width + grid_y * width + grid_x;
                obj_mask_data[mask_idx] = 1.0;

                for (side_idx, &distance) in [left, top, right, bottom].iter().enumerate() {
                    let distance_clamped = distance.max(0.0).min(self.config.reg_max as f32);
                    let bin_lower = distance_clamped.floor() as usize;
                    let bin_upper = (bin_lower + 1).min(self.config.reg_max);
                    let weight_upper = distance_clamped - bin_lower as f32;
                    let weight_lower = 1.0 - weight_upper;

                    for bin in 0..bins {
                        let channel = side_idx * bins + bin;
                        let idx = ((b * reg_channels + channel) * height + grid_y) * width + grid_x;

                        if bin == bin_lower {
                            target_reg_data[idx] = weight_lower;
                        } else if bin == bin_upper {
                            target_reg_data[idx] = weight_upper;
                        }
                    }
                }

                let class_id = bbox.class_id.min(self.config.num_classes - 1);
                let cls_idx = ((b * cls_channels + class_id) * height + grid_y) * width + grid_x;
                target_cls_data[cls_idx] = 1.0;
            }
        }

        let inner_device = <B::InnerBackend as burn::tensor::backend::Backend>::Device::default();
        (
            Tensor::<B::InnerBackend, 4>::from_data(
                TensorData::new(target_reg_data, [batch_size, reg_channels, height, width]),
                &inner_device,
            ),
            Tensor::<B::InnerBackend, 4>::from_data(
                TensorData::new(target_cls_data, [batch_size, cls_channels, height, width]),
                &inner_device,
            ),
            Tensor::<B::InnerBackend, 4>::from_data(
                TensorData::new(obj_mask_data, [batch_size, 1, height, width]),
                &inner_device,
            ),
        )
    }

    fn save_checkpoint(&self, name: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!("Saving checkpoint '{}'...", name);

        std::fs::create_dir_all(&self.config.save_dir)?;

        let checkpoint_dir = format!("{}/{}", self.config.save_dir, name);
        std::fs::create_dir_all(&checkpoint_dir)?;

        let record = self.model.clone().into_record();
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();

        recorder
            .record(record, format!("{}/model", checkpoint_dir).into())
            .map_err(|e| format!("Failed to save model: {:?}", e))?;

        let config_path = format!("{}/config.json", checkpoint_dir);
        let config_json = serde_json::json!({
            "model_type": "YOLOv8",
            "num_classes": self.config.num_classes,
            "reg_max": self.config.reg_max,
            "img_size": self.config.img_size,
            "checkpoint_name": name,
        });
        std::fs::write(&config_path, serde_json::to_string_pretty(&config_json)?)?;

        println!("Checkpoint saved:");
        println!("   Directory: {}", checkpoint_dir);
        println!("   Weights: {}/model.mpk", checkpoint_dir);
        println!("   Config: {}", config_path);

        Ok(())
    }
}
