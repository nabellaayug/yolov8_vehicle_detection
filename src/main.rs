use burn::backend::{Autodiff, NdArray};
use burn::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::backend::AutodiffBackend;

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use yolov8_detection::training::{
    Trainer,
    TrainingConfig,
    TrainingMetrics,
    TrainingState,
};

use yolov8_detection::dataset::{
    DatasetConfig,
};

use yolov8_detection::gui::run_gui;
use yolov8_detection::YOLOLoss;

type BackendType = Autodiff<NdArray>;
type DeviceType = <BackendType as Backend>::Device;
type TrainerType = Trainer<BackendType>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("🚀 YOLOv8 Nano Training (Burn + egui)");
    println!("=====================================\n");

    let num_epochs = 5;
    let num_batches = 10;
    let batch_size = 8;
    let num_classes = 1;
    let image_size = 416; // ✅ YOLOv8 Nano standard input

    println!("📊 Training Config:");
    println!("  Num epochs: {}", num_epochs);
    println!("  Batches per epoch: {}", num_batches);
    println!("  Batch size: {}", batch_size);
    println!("  Image size: {}x{}", image_size, image_size);
    println!("  Num classes:  {}\n", num_classes);

    let training_state = Arc::new(Mutex::new(TrainingState::new()));
    training_state.lock().unwrap().start();
    let training_state_clone = Arc::clone(&training_state);

    let training_thread = thread::spawn(move || {
        if let Err(e) = run_training(
            training_state_clone,
            num_epochs,
            num_batches,
            batch_size,
            num_classes,
            image_size,
        ) {
            eprintln!("❌ Training error: {e}");
        }
    });

    println!("📊 Starting GUI...\n");
    run_gui(training_state)?;

    training_thread.join().ok();
    Ok(())
}

fn run_training(
    training_state: Arc<Mutex<TrainingState>>,
    num_epochs: usize,
    num_batches: usize,
    batch_size: usize,
    num_classes: usize,
    image_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {

    let device: DeviceType = DeviceType::default();

    let config = TrainingConfig {
        num_epochs,
        batch_size,
        learning_rate: 1e-3,
        num_classes,
    };

    let mut trainer: TrainerType = Trainer::new(&device, config);

    println!("{:<8} {:<14} {:<14} {:<10}", "Epoch", "Train Loss", "Val Loss", "Progress");
    println!("{}", "=".repeat(55));

    for epoch in 0..num_epochs {
        log::info!("🔄 EPOCH {} START", epoch + 1);
        
        if !training_state.lock().unwrap().is_running.load(std::sync::atomic::Ordering::SeqCst) {
            log::warn!("⏹ Training stopped");
            break;
        }

        while training_state.lock().unwrap().is_paused.load(std::sync::atomic::Ordering::SeqCst) {
            log::info!("⏸ Training paused, waiting...");
            thread::sleep(Duration::from_millis(500));
        }

        let mut train_loss = 0.0;

        log::info!("📦 Training {} batches...", num_batches);

        for batch_idx in 0..num_batches {
            log::info!("  Batch {}/{} - Generating dummy data...", batch_idx + 1, num_batches);
            
            // ✅ Generate images
            let images = Tensor::random([batch_size, 3, image_size, image_size], Distribution::Uniform(0.0, 1.0), &device);
            
            // ✅ Get actual prediction shapes from forward pass
            let (p2, p3, p4) = trainer.model.forward(images.clone());
            log::info!("    Actual pred shapes: p2={:?}, p3={:?}, p4={:?}", 
                p2.dims(), p3.dims(), p4.dims());
            
            // ✅ Generate targets matching actual prediction output shapes
            let t2 = Tensor::random(p2.dims(), Distribution::Uniform(0.0, 1.0), &device);
            let t3 = Tensor::random(p3.dims(), Distribution::Uniform(0.0, 1.0), &device);
            let t4 = Tensor::random(p4.dims(), Distribution::Uniform(0.0, 1.0), &device);

            log::info!("  Batch {}/{} - Forward & backward...", batch_idx + 1, num_batches);
            let loss = trainer.train_step(images, t2, t3, t4);
            log::info!("    Loss: {:.6}", loss);
            
            train_loss += loss;

            training_state.lock().unwrap().update_batch_metrics(TrainingMetrics {
                epoch: epoch as u32 + 1,
                total_epochs: num_epochs as u32,
                train_loss: train_loss / (batch_idx + 1) as f32,
                val_loss: 0.0,
                learning_rate: trainer.config.learning_rate,
                batch_processed: (batch_idx + 1) as u32,
                total_batches: num_batches as u32,
            });

            // ✅ SHORT SLEEP - batch processing is fast with dummy data
            thread::sleep(Duration::from_millis(200));
        }

        train_loss /= num_batches as f32;
        log::info!("✅ Epoch {} train loss: {:.6}", epoch + 1, train_loss);

        log::info!("🔍 Running validation...");
        let val_loss = run_dummy_validation::<BackendType>(
            &device,
            &trainer,
            batch_size,
        );
        log::info!("✅ Epoch {} val loss: {:.6}", epoch + 1, val_loss);

        log::info!("💾 Pushing epoch metrics to history...");
        training_state.lock().unwrap().push_epoch_metrics(TrainingMetrics {
            epoch: epoch as u32 + 1,
            total_epochs: num_epochs as u32,
            train_loss,
            val_loss,
            learning_rate: trainer.config.learning_rate,
            batch_processed: num_batches as u32,
            total_batches: num_batches as u32,
        });

        println!(
            "{:<8} {:<14.6} {:<14.6} {:>6.1}%",
            format!("{}/{}", epoch + 1, num_epochs),
            train_loss,
            val_loss,
            (epoch + 1) as f32 / num_epochs as f32 * 100.0
        );
    }

    log::info!("🏁 Training completed!");
    Ok(())
}

fn generate_dummy_batch<B: AutodiffBackend>(
    device: &B::Device,
    batch_size: usize,
    image_size: usize,
) -> Result<(Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>), Box<dyn std::error::Error>> {
    let images = Tensor::random([batch_size, 3, image_size, image_size], Distribution::Uniform(0.0, 1.0), device);
    let (t2, t3, t4) = generate_target_tensors::<B>(device, batch_size)?;
    Ok((images, t2, t3, t4))
}

fn generate_target_tensors<B: AutodiffBackend>(
    device: &B::Device,
    batch_size: usize,
) -> Result<(Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>), Box<dyn std::error::Error>> {
    // ✅ Match prediction output shapes!
    // num_anchors * (5 + num_classes) = 3 * (5 + 1) = 18... tapi simplified ke 5 (x,y,w,h,conf)
    let pred_channels = 5;  // or 18 if using full anchor format
    
    Ok((
        Tensor::random([batch_size, pred_channels, 80, 80], Distribution::Uniform(0.0, 1.0), device),
        Tensor::random([batch_size, pred_channels, 40, 40], Distribution::Uniform(0.0, 1.0), device),
        Tensor::random([batch_size, pred_channels, 20, 20], Distribution::Uniform(0.0, 1.0), device),
    ))
}

fn run_dummy_validation<B: AutodiffBackend>(
    device: &B::Device,
    trainer: &Trainer<B>,
    batch_size: usize,
) -> f32 {
    let num_val_batches = 3;
    let mut total_loss = 0.0;

    for batch_idx in 0..num_val_batches {
        log::debug!("  Val batch {}/{}", batch_idx + 1, num_val_batches);
        
        if let Ok((images, t2, t3, t4)) = generate_dummy_batch::<B>(device, batch_size, 416) {
            let (p2, p3, p4) = trainer.model.forward(images);
            let loss = YOLOLoss::compute(p2, p3, p4, t2, t3, t4);
            total_loss += loss.into_scalar().to_f32();
        }
    }

    total_loss / num_val_batches as f32
}