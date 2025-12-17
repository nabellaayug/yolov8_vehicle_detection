# 🔥 YOLOv8 Nano - Rust Implementation from Scratch

A complete **YOLOv8 Nano object detection model** implemented in Rust using the **Burn deep learning framework**. This project demonstrates building a production-ready computer vision model from architecture definition to training and inference.

## 📚 Research & References

### Official Papers & Resources

1. **YOLOv8 Paper & Architecture**
   - 📄 [YOLOv8: A Faster and Stronger Real-Time Object Detector](https://arxiv.org/abs/2308.02146)
   - 🔗 Ultralytics Official: https://github.com/ultralytics/ultralytics
   - 📖 YOLOv8 Docs: https://docs.ultralytics.com/

2. **YOLO Evolution Timeline**
   - YOLOv1 (2016): https://arxiv.org/abs/1506.02640
   - YOLOv3 (2018): https://arxiv.org/abs/1804.02767
   - YOLOv5 (2021): https://github.com/ultralytics/yolov5
   - YOLOv8 (2023): Latest iteration with improved speed/accuracy

### Deep Learning Framework References

3. **Burn Framework** (Rust Deep Learning)
   - 🔗 GitHub: https://github.com/burn-rs/burn
   - 📖 Docs: https://burn.dev/
   - 🎓 Tutorial: https://github.com/burn-rs/burn/tree/main/examples

4. **Related Rust ML Projects**
   - https://github.com/huggingface/safetensors (Tensor serialization)
   - https://github.com/ndarray-rs/ndarray (NumPy-like array library)
   - https://github.com/tch-rs/tch-rs (PyTorch bindings for Rust)

### Neural Network Concepts

5. **Key Architectures Used**
   - **Backbone**: CSPDarknet (CSP = Cross Stage Partial)
   - **Neck**: FPN (Feature Pyramid Network)
   - **Head**: Detection head dengan multi-scale predictions
   - Papers:
     - FPN: https://arxiv.org/abs/1612.03144
     - CSPDarknet: https://arxiv.org/abs/1911.11721

### Training Techniques

6. **Loss Functions & Optimization**
   - IoU Loss: https://arxiv.org/abs/1608.01471
   - Focal Loss: https://arxiv.org/abs/1708.02002
   - Mosaic Data Augmentation (YOLOv4): https://arxiv.org/abs/2004.10934

## 🏗️ Architecture Overview

### Model Structure

```
YOLOv8Nano
├── Backbone (Feature Extractor)
│   ├── Stem: Conv(3 -> 32, 3x3, stride=1)
│   ├── Stage1: C2f(32 -> 64)
│   ├── Stage2: C2f(64 -> 128)
│   ├── Stage3: C2f(128 -> 256)
│   └── Stage4: C2f(256 -> 512)
├── Neck (Feature Aggregation)
│   ├── Top-down pathway (upsampling)
│   ├── Bottom-up pathway (downsampling)
│   └── FPN with lateral connections
└── Head (Detection)
    ├── Conv_p2 (128 channels, 80x80)
    ├── Conv_p3 (256 channels, 40x40)
    └── Conv_p4 (512 channels, 20x20)
```

### Key Components

**C2f Block (Concatenate Fusion)**
- Inspired by CSPDarknet
- Two branches: conv + bottleneck modules
- Concatenated output for richer feature representation
- File: `src/model/blocks/c2f.rs`

**Bottleneck Module**
- Residual connection with hidden dimension reduction
- Structure: conv(in -> hidden) -> conv(hidden -> out) + residual
- Improves gradient flow during training

**FPN Neck**
- Multi-scale feature fusion
- Top-down: high-level features propagate to low-level
- Bottom-up: combines all scales for robust detection
- File: `src/model/neck.rs`

## 📊 Dataset Structure

```
data/
├── raw/
│   ├── images/
│   │   ├── train/      (878 images)
│   │   ├── val/        (250 images)
│   │   ├── test/       (126 images)
│   └── labels/
│       ├── train/      (annotations)
│       ├── val/        (annotations)
│       ├── test/       (annotations)
```

**Supported Formats:**
- Images: JPG, PNG
- Labels: YOLO format (class_id x_center y_center width height)

## 🚀 Installation & Setup

### Prerequisites
- Rust 1.70+ (https://rustup.rs/)
- Cargo (comes with Rust)

### Dependencies
```toml
[dependencies]
burn = { version = "0.19", features = ["autodiff", "ndarray"] }
burn-tch-backend = "0.19"  # Optional: for GPU support
egui = "0.27"              # GUI
egui_plot = "0.27"         # Plotting
eframe = "0.27"            # Window framework
image = "0.24"             # Image loading
rand = "0.8"               # Random numbers
log = "0.4"                # Logging
env_logger = "0.10"        # Logger initialization
```

### Build Instructions
```bash
# Clone repository
git clone <your-repo>
cd yolov8_detection

# Build project
cargo build --release

# Run training
cargo run --release

# Run with optimizations
RUST_LOG=info cargo run --release
```

## Training Pipeline

### Data Loading
- **Real Images**: Load from `data/raw/images/{train,val,test}/`
- **Fallback**: Dummy tensor generation if images fail
- **Preprocessing**: Resize to 224x224, normalize to [0, 1]
- File: `src/main.rs:load_batch_images()`

### Training Loop
```rust
for epoch in 0..num_epochs {
    for batch in dataset.train_batches {
        // Forward pass
        let (p2, p3, p4) = model.forward(images);
        
        // Compute loss
        let loss = YOLOLoss::compute(p2, p3, p4, targets);
        
        // Backward pass & optimization
        trainer.train_step(images, targets);
        
        // Update metrics
        state.update_batch_metrics(metrics);
    }
    
    // Validation
    val_loss = validate(model, val_dataset);
    
    // Early stopping check
    if val_loss < best_val_loss {
        save_weights(model);
    } else {
        patience_counter += 1;
        if patience_counter >= patience {
            break;  // Early stopping
        }
    }
}
```

### Loss Function
Currently: **MSE Loss (simplified)**
```rust
loss = mean((pred - target)^2)
```

Recommended improvements:
- IoU Loss for bounding boxes
- Focal Loss for class imbalance
- Weighted sum of regression + classification loss

File: `src/model/loss.rs`

## Key Implementation Details

### 1. Backbone Architecture (`src/model/backbone.rs`)
- **Input**: 320x320 RGB images
- **Output**: Multi-scale features (p2, p3, p4)
  - p2: 52x52x128 (high resolution, small objects)
  - p3: 26x26x256 (medium scale)
  - p4: 13x13x512 (low resolution, large objects)

### 2. Neck/FPN (`src/model/neck.rs`)
- Upsamples low-res features for fusion
- Downsamples high-res features
- Concatenates aligned features
- Maintains channel consistency

### 3. Detection Head (`src/model/heads.rs`)
- 3 parallel prediction branches
- Output: (batch, channels=5, height, width)
  - 5 = [x, y, w, h, confidence]
  - Expand to (batch, 18, h, w) for 3 anchors + class

### 4. Training State Management (`src/training/state.rs`)
- Thread-safe metrics sharing
- Per-batch metric updates (UI responsiveness)
- Per-epoch history for plotting

### 5. GUI Dashboard (`src/gui/mod.rs`)
- **4 Tabs**:
  1. Training Monitor (live loss plot)
  2. Settings (early stopping config)
  3. Weights (save/load checkpoints)
  4. Test (prediction viewer)
- Real-time metrics visualization
- Controls: Start/Pause/Resume/Stop

## Configuration

Edit `src/main.rs` to adjust:

```rust
let num_epochs = 10;           // Training epochs
let num_batches = 20;          // Batches per epoch
let batch_size = 4;            // Images per batch
let num_classes = 1;           // Number of object classes
let image_size = 224;          // Input size (smaller = faster)
```

**Performance Tips:**
- **Smaller image_size** = Faster training (224 vs 416)
- **Smaller batch_size** = Less memory, but more noisy gradients
- **CPU vs GPU**: Currently CPU (NdArray). For GPU: use `burn-tch-backend` with CUDA

## 📁 Project Structure

```
yolov8_detection/
├── src/
│   ├── main.rs                 # Training entry point
│   ├── lib.rs                  # Library exports
│   ├── model/
│   │   ├── mod.rs              # Model definition
│   │   ├── backbone.rs         # Feature extractor
│   │   ├── neck.rs             # FPN aggregation
│   │   ├── heads.rs            # Detection heads
│   │   ├── loss.rs             # Loss computation
│   │   ├── blocks/
│   │   │   ├── conv.rs         # Conv + BatchNorm + ReLU
│   │   │   ├── c2f.rs          # C2f block
│   │   │   └── mod.rs
│   │   └── yolo.rs             # Main model
│   ├── training/
│   │   ├── mod.rs              # Trainer struct
│   │   └── state.rs            # TrainingState
│   ├── dataset/
│   │   └── mod.rs              # Dataset utilities
│   ├── gui/
│   │   └── mod.rs              # egui dashboard
│   └── lib.rs
├── Cargo.toml                  # Dependencies
├── Cargo.lock                  # Lock file
└── README.md                   # This file
```

## Learning Resources

### YouTube Tutorials
- **YOLO Series Explanation**: 
  - https://www.youtube.com/watch?v=eTDcoeqB1ZA (Object Detection Basics)
  - https://www.youtube.com/watch?v=10joRJx39Ns (YOLOv8 Overview)

- **Rust + Burn Framework**:
  - https://www.youtube.com/watch?v=nZD7VVKrrqE (Burn Tutorial)

### Blog Posts & Articles
- Medium: "Implementing Object Detection with YOLOv8" series
- Towards Data Science: YOLOv8 architecture breakdown
- Hugging Face: Model architecture patterns in Rust

## Future Enhancements

### Short-term
- [ ] Implement real IoU loss function
- [ ] Add data augmentation (Mosaic, mixup)
- [ ] Model weight serialization (save/load)
- [ ] GPU backend support (Wgpu, CUDA)
- [ ] Real prediction inference

### Medium-term
- [ ] Anchor-free detection head
- [ ] Multi-class support
- [ ] Evaluation metrics (mAP, precision, recall)
- [ ] Export to ONNX format

### Long-term
- [ ] Quantization for edge deployment
- [ ] Knowledge distillation
- [ ] Neural Architecture Search (NAS)
- [ ] Multi-GPU distributed training

## Acknowledgments

- **Ultralytics** for YOLOv8 architecture
- **Burn Framework** team for excellent Rust ML library
- **egui** community for reactive UI framework
