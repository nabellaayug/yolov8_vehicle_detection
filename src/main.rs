use burn::backend::NdArray;
use burn::prelude::*;
use burn::tensor::Distribution;

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
    RoadVehicleDataset,
    DatasetConfig,
};

use yolov8_detection::gui::run_gui;
use yolov8_detection::YOLOLoss;

// ======================================================
// 🔧 Backend aliases (PENTING)
// ======================================================
type BackendType = NdArray;
type DeviceType = <BackendType as Backend>::Device;
type TrainerType = Trainer<BackendType>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("🚀 YOLOv8 Nano Training (Burn + egui)");
    println!("====================================\n");

    // --------------------------------------------------
    // Dataset
    // --------------------------------------------------
    let mut dataset = RoadVehicleDataset::new(DatasetConfig {
        data_path: "./data/processed".to_string(),
        image_size: 640,
        batch_size: 8,
        train_split: 0.7,
        val_split: 0.15,
        test_split: 0.15,
    });

    println!("📂 Loading dataset...");
    if let Err(e) = dataset.load_from_folder("data/raw") {
        eprintln!("⚠️ Dataset load failed: {e}");
        println!("⚠️ Using dummy data\n");
    } else {
        println!("✅ Dataset loaded\n");
    }

    let train_images = dataset.get_train_images();
    let val_images = dataset.get_val_images();
    let test_images = dataset.get_test_images();
    let num_classes = dataset.num_classes();

    // --------------------------------------------------
    // Shared training state (GUI <-> training)
    // --------------------------------------------------
    let training_state = Arc::new(Mutex::new(TrainingState::new()));
    let training_state_clone = Arc::clone(&training_state);

    // --------------------------------------------------
    // Spawn training thread
    // --------------------------------------------------
    let training_thread = thread::spawn(move || {
        if let Err(e) = run_training(
            training_state_clone,
            train_images,
            val_images,
            test_images,
            num_classes,
        ) {
            eprintln!("❌ Training error: {e}");
        }
    });

    // --------------------------------------------------
    // Run GUI (MAIN THREAD)
    // --------------------------------------------------
    println!("📊 Starting GUI...\n");
    run_gui(training_state)?;

    training_thread.join().ok();
    Ok(())
}

// ======================================================
// 🧠 Training loop
// ======================================================
fn run_training(
    training_state: Arc<Mutex<TrainingState>>,
    train_images: Vec<String>,
    val_images: Vec<String>,
    _test_images: Vec<String>,
    num_classes: usize,
) -> Result<(), Box<dyn std::error::Error>> {

    // --------------------------------------------------
    // Device (CPU – NdArray)
    // --------------------------------------------------
    let device: DeviceType = DeviceType::default();

    // --------------------------------------------------
    // Training config
    // --------------------------------------------------
    let config = TrainingConfig {
        num_epochs: 10,
        batch_size: 8,
        learning_rate: 1e-3,
        img_size: 640,
        num_classes,
    };

    println!("📊 Training Config:");
    println!("  Epochs       : {}", config.num_epochs);
    println!("  Batch Size   : {}", config.batch_size);
    println!("  LearningRate : {}", config.learning_rate);
    println!("  Image Size   : {}", config.img_size);
    println!("  Num Classes  : {}\n", config.num_classes);

    // --------------------------------------------------
    // ✅ FIX UTAMA: Trainer type eksplisit
    // --------------------------------------------------
    let mut trainer: TrainerType =
        Trainer::new(&device, config);

    println!(
        "{:<8} {:<14} {:<14} {:<10}",
        "Epoch", "Train Loss", "Val Loss", "Progress"
    );
    println!("{}", "=".repeat(55));

    // --------------------------------------------------
    // Epoch loop
    // --------------------------------------------------
    for epoch in 0..trainer.config.num_epochs {

        // Stop training
        {
            let state = training_state.lock().unwrap();
            if !state.is_running.load(std::sync::atomic::Ordering::SeqCst) {
                println!("⏹ Training stopped");
                break;
            }
        }

        // Pause training
        while training_state
            .lock()
            .unwrap()
            .is_paused
            .load(std::sync::atomic::Ordering::SeqCst)
        {
            thread::sleep(Duration::from_millis(100));
        }

        let batch_size = trainer.config.batch_size;
        let num_batches = (train_images.len() / batch_size).max(1);
        let mut train_loss = 0.0;

        // --------------------------------------------------
        // Batch loop
        // --------------------------------------------------
        for batch_idx in 0..num_batches {
            let (images, t2, t3, t4) =
                load_batch::<BackendType>(&device, batch_size)?;

            let loss = trainer.train_step(images, t2, t3, t4);
            train_loss += loss;

            training_state.lock().unwrap().update_metrics(
                TrainingMetrics {
                    epoch: epoch as u32 + 1,
                    total_epochs: trainer.config.num_epochs as u32,
                    train_loss: train_loss / (batch_idx + 1) as f32,
                    val_loss: 0.0,
                    learning_rate: trainer.config.learning_rate,
                    batch_processed: (batch_idx + 1) as u32,
                    total_batches: num_batches as u32,
                }
            );

            thread::sleep(Duration::from_millis(30));
        }

        train_loss /= num_batches as f32;

        // --------------------------------------------------
        // Validation
        // --------------------------------------------------
        let val_loss = run_validation::<BackendType>(
            &device,
            &trainer,
            &val_images,
            batch_size,
        );

        training_state.lock().unwrap().update_metrics(
            TrainingMetrics {
                epoch: epoch as u32 + 1,
                total_epochs: trainer.config.num_epochs as u32,
                train_loss,
                val_loss,
                learning_rate: trainer.config.learning_rate,
                batch_processed: num_batches as u32,
                total_batches: num_batches as u32,
            }
        );

        println!(
            "{:<8} {:<14.6} {:<14.6} {:>6.1}%",
            format!("{}/{}", epoch + 1, trainer.config.num_epochs),
            train_loss,
            val_loss,
            (epoch + 1) as f32 / trainer.config.num_epochs as f32 * 100.0
        );
    }

    println!("\n✨ Training finished");
    Ok(())
}

// ======================================================
// 📦 Dummy batch loader
// ======================================================
fn load_batch<B: Backend>(
    device: &B::Device,
    batch_size: usize,
) -> Result<
    (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>),
    Box<dyn std::error::Error>,
> {
    Ok((
        Tensor::random(
            [batch_size, 3, 640, 640],
            Distribution::Uniform(0.0, 1.0),
            device,
        ),
        Tensor::random(
            [batch_size, 255, 160, 160],
            Distribution::Uniform(0.0, 1.0),
            device,
        ),
        Tensor::random(
            [batch_size, 255, 80, 80],
            Distribution::Uniform(0.0, 1.0),
            device,
        ),
        Tensor::random(
            [batch_size, 255, 40, 40],
            Distribution::Uniform(0.0, 1.0),
            device,
        ),
    ))
}

// ======================================================
// 📊 Validation loop
// ======================================================
fn run_validation<B: Backend>(
    device: &B::Device,
    trainer: &Trainer<B>,
    val_images: &[String],
    batch_size: usize,
) -> f32 {
    let num_batches = (val_images.len() / batch_size).max(1);
    let mut total_loss = 0.0;

    for _ in 0..num_batches {
        if let Ok((images, t2, t3, t4)) =
            load_batch::<B>(device, batch_size)
        {
            let (p2, p3, p4) = trainer.model.forward(images);
            let loss = YOLOLoss::compute(p2, p3, p4, t2, t3, t4);
            total_loss += loss.into_scalar().to_f32();
        }
    }

    total_loss / num_batches as f32
}
