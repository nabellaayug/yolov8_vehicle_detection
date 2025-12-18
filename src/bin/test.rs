use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::record::{Recorder, BinFileRecorder, FullPrecisionSettings};
use burn::prelude::*;
use clap::Parser;
use image::{DynamicImage, GenericImageView};

use yolov8_detection::model::YOLOv8;

#[derive(Parser, Debug)]
#[command(author, version, about = "YOLOv8 Inference - Test on images")]
struct Args {
    /// Path to input image
    #[arg(short, long)]
    image: String,
    
    /// Path to model checkpoint directory (e.g., runs/train/best)
    #[arg(short, long, default_value = "runs/train/best")]
    weights: String,
    
    /// Confidence threshold
    #[arg(short, long, default_value_t = 0.50)]
    conf: f32,
    
    /// IoU threshold for NMS
    #[arg(long, default_value_t = 0.45)]
    iou: f32,
    
    /// Save output image with bounding boxes
    #[arg(short, long)]
    save: bool,
}

// Class names for Cars Detection dataset
const CLASS_NAMES: &[&str] = &["Ambulance", "Bus", "Car", "Motorcycle", "Truck"];

#[derive(Debug, Clone)]
struct Detection {
    x: f32,      // Center x
    y: f32,      // Center y
    w: f32,      // Width
    h: f32,      // Height
    confidence: f32,
    class_id: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    
    println!("YOLOv8 Car Detection Inference");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Input: {}", args.image);
    println!("Weights: {}", args.weights);
    println!("Confidence: {:.2}", args.conf);
    println!("IoU: {:.2}", args.iou);
    println!();
    
    // Setup CPU backend (NdArray)
    type MyBackend = NdArray;
    let device = NdArrayDevice::default();
    
    // Load config
    println!("Loading model config...");
    let config_path = format!("{}/config.json", args.weights);
    let config_content = std::fs::read_to_string(&config_path)?;
    let config: serde_json::Value = serde_json::from_str(&config_content)?;
    
    let num_classes = config["num_classes"].as_u64().unwrap_or(5) as usize;
    let reg_max = config["reg_max"].as_u64().unwrap_or(16) as usize;
    let img_size = config["img_size"].as_u64().unwrap_or(640) as usize;
    
    println!("  Classes: {}", num_classes);
    println!("  Reg Max: {}", reg_max);
    println!("  Image Size: {}", img_size);
    println!();
    
    // Create model
    println!("Creating model...");
    let mut model = YOLOv8::<MyBackend>::new(&device, num_classes, reg_max);
    
    // Load weights
    println!("Loading checkpoint...");
    let weights_path = format!("{}/model", args.weights);
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();

    let record = recorder.load(weights_path.into(), &device)
        .map_err(|e| format!("Failed to load weights: {:?}", e))?;
    
    model = model.load_record(record);
    println!("Weights loaded successfully!");
    println!();
    
    // Load and preprocess image
    println!("Loading image...");
    let img = image::open(&args.image)?;
    let (orig_w, orig_h) = img.dimensions();
    println!("Original size: {}x{}", orig_w, orig_h);
    
    // Resize to model input size
    let img_resized = img.resize_exact(
        img_size as u32, 
        img_size as u32, 
        image::imageops::FilterType::Lanczos3
    );
    
    // Convert to tensor [1, 3, H, W] normalized [0, 1]
    let img_tensor = image_to_tensor(&img_resized, &device);
    println!("Preprocessed to {}x{}", img_size, img_size);
    println!();
    
    // Run inference
    println!("Running inference...");
    let start = std::time::Instant::now();
    
    let ((pred_reg_p2, pred_cls_p2), (pred_reg_p3, pred_cls_p3), (pred_reg_p4, pred_cls_p4)) 
        = model.forward(img_tensor);
    
    let inference_time = start.elapsed();
    println!("Inference completed in {:.2}ms", inference_time.as_millis());
    println!();
    
    // Post-process predictions
    println!("Post-processing...");
    let mut detections = Vec::new();
    
    // Process each scale
    detections.extend(decode_predictions(&pred_reg_p2, &pred_cls_p2, 8, img_size, args.conf));
    detections.extend(decode_predictions(&pred_reg_p3, &pred_cls_p3, 16, img_size, args.conf));
    detections.extend(decode_predictions(&pred_reg_p4, &pred_cls_p4, 32, img_size, args.conf));
    
    println!("Found {} raw detections", detections.len());
    
    // Apply NMS
    detections = apply_nms(detections, args.iou);
    println!("After NMS: {} detections", detections.len());
    println!();
    
    // Print detections
    if detections.is_empty() {
        println!("No objects detected!");
        println!("   Try lowering confidence threshold with --conf 0.1");
    } else {
        println!("Detected {} objects:", detections.len());
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        for (i, det) in detections.iter().enumerate() {
            let class_name = if det.class_id < CLASS_NAMES.len() {
                CLASS_NAMES[det.class_id]
            } else {
                "Unknown"
            };
            
            // Convert back to original image coordinates
            let x1 = ((det.x - det.w / 2.0) * orig_w as f32 / img_size as f32).max(0.0);
            let y1 = ((det.y - det.h / 2.0) * orig_h as f32 / img_size as f32).max(0.0);
            let x2 = ((det.x + det.w / 2.0) * orig_w as f32 / img_size as f32).min(orig_w as f32);
            let y2 = ((det.y + det.h / 2.0) * orig_h as f32 / img_size as f32).min(orig_h as f32);
            
            println!("{}. {} ({:.1}%)", i + 1, class_name, det.confidence * 100.0);
            println!("   BBox: [{:.0}, {:.0}, {:.0}, {:.0}] (x1, y1, x2, y2)", x1, y1, x2, y2);
            println!("   Size: {:.0}x{:.0}", x2 - x1, y2 - y1);
        }
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }
    println!();
    
    // Optionally save image with boxes
    if args.save && !detections.is_empty() {
        println!("Saving output image...");
        save_detections_image(&img, &detections, img_size, orig_w, orig_h, &args.image)?;
    }
    
    println!("Done!");
    
    Ok(())
}

/// Convert image to tensor [1, 3, H, W] normalized to [0, 1]
fn image_to_tensor(img: &DynamicImage, device: &NdArrayDevice) -> Tensor<NdArray, 4> {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    
    let mut data = vec![0.0f32; 3 * height as usize * width as usize];
    
    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x, y);
            let idx = (y * width + x) as usize;
            
            // CHW format: [R_all, G_all, B_all]
            data[idx] = pixel[0] as f32 / 255.0;  // R channel
            data[height as usize * width as usize + idx] = pixel[1] as f32 / 255.0;  // G channel
            data[2 * height as usize * width as usize + idx] = pixel[2] as f32 / 255.0;  // B channel
        }
    }
    
    Tensor::<NdArray, 4>::from_data(
        TensorData::new(data, [1, 3, height as usize, width as usize]),
        device
    )
}

/// Decode predictions from one scale
fn decode_predictions(
    pred_reg: &Tensor<NdArray, 4>,
    pred_cls: &Tensor<NdArray, 4>,
    stride: usize,
    img_size: usize,
    conf_threshold: f32,
) -> Vec<Detection> {
    let mut detections = Vec::new();
    
    let [_batch, _reg_ch, height, width] = pred_reg.dims();
    let [_batch, num_classes, _h, _w] = pred_cls.dims();
    //Get data
    let reg_data_raw = pred_reg.clone().into_data();
    let cls_data_raw = pred_cls.clone().into_data();
    
    // Convert to Vec<f32>
    let reg_data: Vec<f32> = reg_data_raw.iter::<f32>().collect();
    let cls_data: Vec<f32> = cls_data_raw.iter::<f32>().collect();
    
    for y in 0..height {
        for x in 0..width {
            // Get class scores
            let mut max_score = 0.0f32;
            let mut max_class = 0;
            
            for c in 0..num_classes {
                let idx = c * height * width + y * width + x;
                let score = 1.0 / (1.0 + (-cls_data[idx]).exp()); // Sigmoid
                
                if score > max_score {
                    max_score = score;
                    max_class = c;
                }
            }
            
            // Filter by confidence
            if max_score < conf_threshold {
                continue;
            }
            
            // Decode bounding box (simplified - use first 4 channels as left, top, right, bottom)
            let left = reg_data[0 * height * width + y * width + x];
            let top = reg_data[1 * height * width + y * width + x];
            let right = reg_data[2 * height * width + y * width + x];
            let bottom = reg_data[3 * height * width + y * width + x];
            
            // Convert to center + size format
            let center_x = (x as f32 + 0.5) * stride as f32;
            let center_y = (y as f32 + 0.5) * stride as f32;
            
            let box_w = (left + right) * stride as f32;
            let box_h = (top + bottom) * stride as f32;
            
            // Skip if box is too small or invalid
            if box_w <= 0.0 || box_h <= 0.0 || box_w > img_size as f32 * 2.0 || box_h > img_size as f32 * 2.0 {
                continue;
            }
            
            detections.push(Detection {
                x: center_x,
                y: center_y,
                w: box_w,
                h: box_h,
                confidence: max_score,
                class_id: max_class,
            });
        }
    }
    
    detections
}

/// Apply Non-Maximum Suppression
fn apply_nms(mut detections: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    // Sort by confidence (highest first)
    detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    
    let mut keep = Vec::new();
    
    while !detections.is_empty() {
        let best = detections.remove(0);
        keep.push(best.clone());
        
        detections.retain(|det| {
            // Only compare same class
            if det.class_id != best.class_id {
                return true;
            }
            
            let iou = compute_iou(&best, det);
            iou < iou_threshold
        });
    }
    
    keep
}

/// Compute IoU between two detections
fn compute_iou(det1: &Detection, det2: &Detection) -> f32 {
    let x1_min = det1.x - det1.w / 2.0;
    let y1_min = det1.y - det1.h / 2.0;
    let x1_max = det1.x + det1.w / 2.0;
    let y1_max = det1.y + det1.h / 2.0;
    
    let x2_min = det2.x - det2.w / 2.0;
    let y2_min = det2.y - det2.h / 2.0;
    let x2_max = det2.x + det2.w / 2.0;
    let y2_max = det2.y + det2.h / 2.0;
    
    let inter_x = (x1_max.min(x2_max) - x1_min.max(x2_min)).max(0.0);
    let inter_y = (y1_max.min(y2_max) - y1_min.max(y2_min)).max(0.0);
    
    let intersection = inter_x * inter_y;
    let area1 = det1.w * det1.h;
    let area2 = det2.w * det2.h;
    let union = area1 + area2 - intersection;
    
    intersection / union.max(1e-6)
}

/// Save image with detections (simple text output)
fn save_detections_image(
    img: &DynamicImage,
    detections: &[Detection],
    img_size: usize,
    orig_w: u32,
    orig_h: u32,
    input_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create output filename
    let input_path_obj = std::path::Path::new(input_path);
    let stem = input_path_obj.file_stem().unwrap().to_str().unwrap();
    let output_path = format!("{}_detections.txt", stem);
    
    // Write detections to text file
    let mut output = String::new();
    output.push_str(&format!("Image: {} ({}x{})\n", input_path, orig_w, orig_h));
    output.push_str(&format!("Detections: {}\n\n", detections.len()));
    
    for (i, det) in detections.iter().enumerate() {
        let class_name = if det.class_id < CLASS_NAMES.len() {
            CLASS_NAMES[det.class_id]
        } else {
            "Unknown"
        };
        
        let x1 = ((det.x - det.w / 2.0) * orig_w as f32 / img_size as f32).max(0.0);
        let y1 = ((det.y - det.h / 2.0) * orig_h as f32 / img_size as f32).max(0.0);
        let x2 = ((det.x + det.w / 2.0) * orig_w as f32 / img_size as f32).min(orig_w as f32);
        let y2 = ((det.y + det.h / 2.0) * orig_h as f32 / img_size as f32).min(orig_h as f32);
        
        output.push_str(&format!(
            "{}. {} {:.1}%\n   BBox: [{:.0}, {:.0}, {:.0}, {:.0}]\n\n",
            i + 1, class_name, det.confidence * 100.0, x1, y1, x2, y2
        ));
    }
    
    std::fs::write(&output_path, output)?;
    println!(" Saved detections to: {}", output_path);
    
    Ok(())
}