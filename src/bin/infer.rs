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
    
    println!("YOLOv8 Inference (CPU)");
    println!("Image: {}", args.image);
    println!("Weights: {}", args.weights);
    println!("Confidence: {}", args.conf);
    println!("IoU: {}", args.iou);
    println!();
    
    // Setup CPU backend (NdArray)
    type MyBackend = NdArray;
    let device = NdArrayDevice::default();
    
    println!("Creating model...");
    let _model = YOLOv8::<MyBackend>::new(&device, args.classes, 16);
    
    println!("Model created (weights loading not yet implemented)");
    
    println!(" Loading image: {}", args.image);
    
    println!(" Running inference...");

    let detections: Vec<yolov8_detection::model::nms::Detection> = Vec::new();
    
    println!("Inference completed");
    println!("Found {} objects", detections.len());
    
    if detections.is_empty() {
        println!("No detections found (inference not yet implemented)");
    } else {
        for (i, det) in detections.iter().enumerate() {
            println!("  {}. Class: {}, Confidence: {:.2}%", 
                   i + 1, det.class_id, det.confidence * 100.0);
        }
    }
    
    Ok(())
}