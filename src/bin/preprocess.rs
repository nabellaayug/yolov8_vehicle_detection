use std::path::PathBuf;
use clap::{Parser};

use yolov8_detection::dataset::ImagePreprocessor;

#[derive(Parser, Debug)]
struct Args {
    /// Path ke dataset folder
    #[arg(short, long)]
    dataset_path: String,

    /// Output path untuk preprocessed data
    #[arg(short, long)]
    output_path: String,

    /// Target image size
    #[arg(short, long, default_value = "640")]
    size: u32,

    /// Number of workers
    #[arg(short, long, default_value = "4")]
    workers: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .init();

    let args = Args::parse();

    println!("🚀 YOLOv8 Image Preprocessor");
    println!("===============================\n");

    println!("📂 Dataset Path: {}", args.dataset_path);
    println!("📁 Output Path: {}", args.output_path);
    println!("📏 Target Size: {}x{}", args.size, args.size);
    println!("👷 Workers: {}\n", args.workers);

    let preprocessor = ImagePreprocessor::new(PathBuf::from(&args.dataset_path), args.size);

    // Load semua images
    println!("🔍 Scanning dataset...");
    let image_paths = preprocessor.load_dataset()?;
    println!("✅ Found {} images\n", image_paths.len());

    if image_paths.is_empty() {
        println!("⚠️  No images found in dataset");
        return Ok(());
    }

    // Preprocess dengan multi-threading
    println!("⚙️  Starting preprocessing...\n");

    let total = image_paths.len();
    let mut processed = 0;
    let mut errors = 0;

    for (idx, image_path) in image_paths.iter().enumerate() {
        match preprocessor.preprocess_image(image_path) {
            Ok(image_data) => {
                match preprocessor.save_preprocessed(&image_data, PathBuf::from(&args.output_path).as_path()) {
                    Ok(_) => {
                        processed += 1;
                    }
                    Err(e) => {
                        eprintln!("❌ Error saving {}: {}", image_path.display(), e);
                        errors += 1;
                    }
                }
            }
            Err(e) => {
                eprintln!("❌ Error processing {}: {}", image_path.display(), e);
                errors += 1;
            }
        }

        // Progress
        if (idx + 1) % 10 == 0 || idx + 1 == total {
            let percent = ((idx + 1) as f32 / total as f32) * 100.0;
            println!("Progress: {}/{} ({:.1}%) | Errors: {}", idx + 1, total, percent, errors);
        }
    }

    println!("\n✨ Preprocessing completed!");
    println!("📊 Summary:");
    println!("  - Processed: {}", processed);
    println!("  - Errors: {}", errors);
    println!("  - Success Rate: {:.2}%", (processed as f32 / total as f32) * 100.0);

    Ok(())
}