use burn::backend::{Autodiff, NdArray};
use yolov8_detection::training::{Trainer, TrainingConfig};
use yolov8_detection::data::dataset::reorganize_dataset;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🦀 YOLOv8 Rust Training (CPU)");
    println!("==============================\n");
    
    // Setup CPU backend (NdArray)
    type MyBackend = NdArray;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    
    let device = Default::default();
    
    // Load or create config
    let config = if Path::new("configs/train_config.yaml").exists() {
        println!("📄 Loading config from configs/train_config.yaml");
        TrainingConfig::from_yaml("configs/train_config.yaml")?
    } else {
        let config = TrainingConfig::default();
        std::fs::create_dir_all("configs")?;
        config.save("configs/train_config.yaml")?;
        println!("✅ Created default config at configs/train_config.yaml");
        config
    };
    
    println!("\n📊 Training Configuration:");
    println!("  Dataset YAML: {}", config.data_yaml);
    println!("  Epochs: {}", config.epochs);
    println!("  Batch size: {}", config.batch_size);
    println!("  Learning rate: {}", config.learning_rate);
    println!("  Image size: {}x{}", config.img_size, config.img_size);
    println!("  Save dir: {}", config.save_dir);
    println!();
    
    // Verify and prepare dataset
    println!("🔍 Verifying dataset structure...");
    
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
        println!("\n⚠️  Old dataset structure detected!");
        println!("🔄 Auto-reorganizing dataset...\n");
        reorganize_dataset(dataset_dir)?;
        println!();
    } else if !new_structure {
        eprintln!("\n❌ Error: Dataset structure not found!");
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
    
    println!("  ✅ Dataset ready");
    println!();
    
    // Create trainer
    println!("🚀 Initializing trainer...");
    let mut trainer = Trainer::<MyAutodiffBackend>::new(config, device);
    println!("  ✅ Trainer initialized\n");
    
    // Start training
    println!("🚀 Starting training...\n");
    match trainer.train() {
        Ok(_) => {
            println!("\n✅ Training completed successfully!");
            println!("📁 Checkpoints saved in 'runs/train/' directory");
            Ok(())
        }
        Err(e) => {
            eprintln!("\n❌ Training failed: {}", e);
            Err(e)
        }
    }
}