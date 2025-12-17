use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use clap::Parser;

use yolov8_detection::model::YOLOv8;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to input image
    #[arg(short, long)]
    image: String,
    
    /// Path to model weights
    #[arg(short, long, default_value = "runs/train/best.bin")]
    weights: String,
    
    /// Confidence threshold
    #[arg(short, long, default_value_t = 0.25)]
    conf: f32,
    
    /// IoU threshold for NMS
    #[arg(long, default_value_t = 0.45)]
    iou: f32,
    
    /// Number of classes
    #[arg(long, default_value_t = 80)]
    classes: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    
    println!("🔍 YOLOv8 Inference (CPU)");
    println!("=========================");
    println!("Image: {}", args.image);
    println!("Weights: {}", args.weights);
    println!("Confidence: {}", args.conf);
    println!("IoU: {}", args.iou);
    println!();
    
    // Setup CPU backend (NdArray)
    type MyBackend = NdArray;
    let device = NdArrayDevice::default();
    
    println!("🔧 Creating model...");
    let _model = YOLOv8::<MyBackend>::new(&device, args.classes, 16);
    
    // TODO: Load weights
    // _model.load_weights(&args.weights)?;
    println!("✅ Model created (weights loading not yet implemented)");
    
    // TODO: Load and preprocess image
    println!("📷 Loading image: {}", args.image);
    // let img_tensor = load_image(&args.image, &device)?;
    
    // TODO: Run inference
    println!("🚀 Running inference...");
    // let detections = model.predict(img_tensor, args.conf, args.iou);
    
    // ✅ FIX #2: Placeholder untuk detections sampai inference di-implement
    let detections: Vec<yolov8_detection::model::nms::Detection> = Vec::new();
    
    println!("✅ Inference completed");
    println!("📊 Found {} objects", detections.len());
    
    // TODO: Visualize results
    // ✅ FIX #3: Safety check untuk loop
    if detections.is_empty() {
        println!("⚠️  No detections found (inference not yet implemented)");
    } else {
        for (i, det) in detections.iter().enumerate() {
            println!("  {}. Class: {}, Confidence: {:.2}%", 
                   i + 1, det.class_id, det.confidence * 100.0);
        }
    }
    
    Ok(())
}