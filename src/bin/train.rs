use burn::backend::{Autodiff, NdArray};
use yolov8_detection::training::{Trainer, TrainingConfig};
use yolov8_detection::data::dataset::reorganize_dataset;
use yolov8_detection::gui::training_gui::{TrainingGui, TrainingState};
use std::path::Path;
use std::thread;
use std::sync::{Arc, Mutex};
use eframe::egui;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("YOLOv8 Rust Training (CPU)");
    
    // Setup CPU backend (NdArray wrapped with Autodiff)
    type MyBackend = NdArray;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    
    let device = <MyAutodiffBackend as burn::tensor::backend::Backend>::Device::default();
    
    // Load or create config
    let config = if Path::new("configs/train_config.yaml").exists() {
        println!("Loading config from configs/train_config.yaml");
        TrainingConfig::from_yaml("configs/train_config.yaml")?
    } else {
        let config = TrainingConfig::default();
        std::fs::create_dir_all("configs")?;
        config.save("configs/train_config.yaml")?;
        println!("Created default config at configs/train_config.yaml");
        config
    };
    
    println!("\nTraining Configuration:");
    println!("  Dataset YAML: {}", config.data_yaml);
    println!("  Epochs: {}", config.epochs);
    println!("  Batch size: {}", config.batch_size);
    println!("  Learning rate: {}", config.learning_rate);
    println!("  Image size: {}x{}", config.img_size, config.img_size);
    println!("  Save dir: {}", config.save_dir);
    println!();
    
    // Verify and prepare dataset
    println!("Verifying dataset structure...");
    
    let dataset_dir = "data/Cars Detection";
    let dataset_yaml_path = format!("{}/data.yaml", dataset_dir);
    
    let yaml_exists = Path::new(&dataset_yaml_path).exists();
    let old_structure = Path::new(dataset_dir).join("train").join("images").exists();
    let new_structure = Path::new(dataset_dir).join("images").join("train").exists()
        && Path::new(dataset_dir).join("labels").join("train").exists();
    
    println!("  Checking dataset at: {}", dataset_dir);
    println!("  YAML exists: {}", yaml_exists);
    println!("  New structure (images/labels): {}", new_structure);
    println!("  Old structure (train/images): {}", old_structure);
    
    // Auto-reorganize if old structure detected
    if old_structure && !new_structure {
        println!("\nOld dataset structure detected!");
        println!("Auto-reorganizing dataset...\n");
        reorganize_dataset(dataset_dir)?;
        println!();
    } else if !new_structure {
        eprintln!("\nError: Dataset structure not found!");
        eprintln!("   Expected: data/Cars Detection/ with this structure:");
        eprintln!("   ├── images/");
        eprintln!("   │   ├── train/");
        eprintln!("   │   ├── test/");
        eprintln!("   │   └── val/");
        eprintln!("   ├── labels/");
        eprintln!("   │   ├── train/");
        eprintln!("   │   ├── test/");
        eprintln!("   │   └── val/");
        eprintln!("   └── data.yaml");
        return Err("Dataset structure not found".into());
    }
    
    println!(" Dataset ready");
    println!();
    
    // Create shared training state
    let training_state = Arc::new(Mutex::new(TrainingState {
        train_losses: Vec::new(),
        val_losses: Vec::new(),
        current_epoch: 0,
        total_epochs: config.epochs,
        learning_rate: config.learning_rate,
        status: "Initializing trainer...".to_string(),
        is_training: true,
    }));
    
    let state_for_training = training_state.clone();
    
    // Start training in background thread
    println!("Starting training thread...");
    thread::spawn(move || {
        println!("Initializing trainer...");
        let mut trainer = Trainer::<MyAutodiffBackend>::new(config.clone(), device);
        println!("Trainer initialized\n");
        
        if let Ok(mut state) = state_for_training.lock() {
            state.status = "Starting training...".to_string();
        }
        
        println!("Starting training...\n");
        
        match trainer.train_with_gui(state_for_training.clone()) {
            Ok(_) => {
                println!("\nTraining completed successfully!");
                println!(" Checkpoints saved in 'runs/train/' directory");
                
                if let Ok(mut state) = state_for_training.lock() {
                    state.is_training = false;
                    state.status = "Training completed successfully!".to_string();
                }
            }
            Err(e) => {
                eprintln!("\n Training failed: {}", e);
                
                if let Ok(mut state) = state_for_training.lock() {
                    state.is_training = false;
                    state.status = format!(" Training failed: {}", e);
                }
            }
        }
    });
    
    // Run GUI in main thread (required for Windows)
    println!(" Launching training monitor GUI in main thread...\n");
    
    let gui = TrainingGui::from_state(training_state);
    
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 700.0])
            .with_title(" YOLOv8 Training Monitor"),
        ..Default::default()
    };
    
    eframe::run_native(
        "YOLOv8 Training Monitor",
        options,
        Box::new(move |_cc| Ok(Box::new(gui))),
    )?;
    
    println!("\n GUI closed. Training may still be running in background.");
    Ok(())
}