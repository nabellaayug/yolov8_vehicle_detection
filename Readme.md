# 🚗 YOLOv8 Object Detection in Rust

A high-performance YOLOv8 object detection implementation in Rust using the Burn deep learning framework. This project focuses on vehicle detection with 5 classes: Ambulance, Bus, Car, Motorcycle, and Truck.

[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![Burn](https://img.shields.io/badge/burn-0.19-blue.svg)](https://burn.dev/)

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Training](#-training)
- [Inference](#-inference)
- [Troubleshooting](#-troubleshooting)
- [Resources](#-resources)

---

## ✨ Features

- ✅ **Pure Rust Implementation** - No Python dependencies
- ✅ **Burn Framework** - Modern deep learning in Rust
- ✅ **Multi-Scale Detection** - 3 detection heads (P2, P3, P4)
- ✅ **Real-time Training GUI** - Live loss visualization with egui
- ✅ **Flexible Backend** - CPU (NdArray) support
- ✅ **Early Stopping** - Prevent overfitting automatically
- ✅ **Checkpoint Management** - Save best and periodic checkpoints
- ✅ **Easy Inference** - Simple CLI for testing on images
- ✅ **YOLO Format Support** - Standard YOLO dataset format


### Software Requirements
- **Rust:** 1.75 or higher
  - Install from [rustup.rs](https://rustup.rs/)
- **Git:** For cloning the repository

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/yolov8-rust-detection.git
cd yolov8-rust-detection
```

### 2. Install Rust Dependencies

```bash
cargo build --release
```

This will download and compile all dependencies (~10-15 minutes first time).

### 3. Verify Installation

```bash
cargo test
```

All tests should pass ✅

---

## 📦 Dataset Preparation

### Option 1: Use Cars Detection Dataset (Recommended)

1. **Download Dataset**
   - Download from [Kaggle](https://www.kaggle.com/datasets/abdallahwagih/cars-detection) 
   - Format: YOLO v8

2. **Extract to Project**
   ```bash
   # Extract to data/ folder
   unzip cars-detection.zip -d data/
   ```

3. **Verify Structure**
   ```bash
   data/Cars Detection/
   ├── images/
   │   ├── train/     # Training images (.jpg, .png)
   │   ├── val/       # Validation images
   │   └── test/      # Test images
   ├── labels/
   │   ├── train/     # Training labels (.txt)
   │   ├── val/       # Validation labels
   │   └── test/      # Test labels
   └── data.yaml      # Dataset config
   ```

4. **Verify data.yaml**
   ```yaml
   # data.yaml
   train: images/train
   val: images/val
   test: images/test
   
   nc: 5  # number of classes
   names: ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']
   ```

### Option 2: Use Your Own Dataset

1. **Organize Your Data** in YOLO format:
   ```
   your_dataset/
   ├── images/train/*.jpg
   ├── images/val/*.jpg
   ├── labels/train/*.txt
   └── labels/val/*.txt
   ```

2. **Label Format** (YOLO format):
   ```
   # Each line: class_id center_x center_y width height (normalized 0-1)
   0 0.5 0.5 0.3 0.4
   2 0.3 0.6 0.2 0.3
   ```

3. **Create data.yaml**:
   ```yaml
   train: images/train
   val: images/val
   nc: 5
   names: ['class1', 'class2', 'class3', 'class4', 'class5']
   ```

4. **Update config.rs**:
   ```rust
   // src/training/config.rs
   data_yaml: "data/your_dataset/data.yaml".to_string(),
   num_classes: 5,  // Your number of classes
   ```

---

## 🎓 Training

### Quick Start

```bash
# Start training with default config
cargo run --release --bin train
```

This will:
- ✅ Load dataset from `data/Cars Detection/`
- ✅ Initialize YOLOv8 model
- ✅ Train for 10 epochs
- ✅ Show real-time GUI with loss plots
- ✅ Save checkpoints to `runs/train/`

### Training Output

```
YOLOv8 Rust Training (CPU)
Dataset loaded:
  Train: 500 images
  Val: 100 images

Epoch [1/10]
  📊 Batch 10: loss=0.7200
  📊 Batch 20: loss=0.6950
🔍 Starting validation with ~10 batches...
  [Val Batch 1] 3 objects → Losses: p2=0.7100, p3=0.6900, p4=0.7000
✅ Validation completed: 10 valid batches
  Train Loss: 0.6700, Val Loss: 0.7100
  ✅ Validation loss improved! Saving best checkpoint...
  💾 Saving checkpoint 'best'...
  ✅ Checkpoint saved
  ⏱️ Epoch time: 245s

Epoch [2/10]
  ...

💾 Saving final checkpoint...
✅ Training completed!
📁 Checkpoints saved in: runs/train
```

### Training GUI

The GUI shows real-time:
- 📈 **Loss plots** (train & validation)
- 🔄 **Current epoch progress**
- ⚡ **Learning rate**
- ⏱️ **Epoch time**
- ⚠️ **Overfitting warnings**

![Training GUI Screenshot](docs/training_gui.png)

### Custom Training Configuration

Edit `configs/train_config.yaml`:

```yaml
# Dataset
data_yaml: "data/Cars Detection/data.yaml"
img_size: 640
num_classes: 5

# Training
epochs: 50                  # Number of epochs
batch_size: 8               # Batch size (increase if you have more RAM)
learning_rate: 0.001        # Learning rate (CRITICAL!)
weight_decay: 0.0005
warmup_epochs: 3

# Model
reg_max: 16

# Loss weights
box_loss_weight: 7.5
cls_loss_weight: 0.5
dfl_loss_weight: 1.5

# Inference
conf_threshold: 0.25        # Confidence threshold
iou_threshold: 0.45         # NMS IoU threshold

# Early stopping
patience: 10                # Stop if no improvement for N epochs
min_delta: 0.001

# Checkpointing
save_dir: "runs/train"
save_interval: 10           # Save every N epochs
```

### Important Training Notes

⚠️ **Learning Rate is CRITICAL!**
- Default: `0.001` ✅

⚠️ **Batch Size**
- Increase if you have more RAM
- Decrease if you get OOM errors
- Default: `1` (safe for 8GB RAM)

⚠️ **Epochs**
- Start with 10-20 for testing
- 50-100 for good results
- Use early stopping to prevent overfitting

---

## 🔮 Inference

### Test on Single Image

```bash
# Basic usage
cargo run --release --bin test -- --image "path/to/car.jpg"

# With custom confidence threshold
cargo run --release --bin test -- --image "test.jpg" --conf 0.15

# Use different checkpoint
cargo run --release --bin test -- \
  --image "test.jpg" \
  --weights "runs/train/epoch_50" \
  --conf 0.25

# Save detections to file
cargo run --release --bin test -- --image "test.jpg" --save
```

### Inference Output

```
🚗 YOLOv8 Car Detection Inference
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📷 Input: test_car.jpg
🔧 Weights: runs/train/best
🎯 Confidence: 0.25
📊 IoU: 0.45

📋 Loading model config...
  ✅ Classes: 5
  ✅ Image Size: 640

🔨 Creating model...
📦 Loading checkpoint...
  ✅ Weights loaded successfully!

🖼️  Loading image...
  ✅ Original size: 1280x720

🔮 Running inference...
  ✅ Inference completed in 245ms

📊 Post-processing...
  ✅ Found 15 raw detections
  ✅ After NMS: 3 detections

🎯 Detected 3 objects:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Car (87.3%)
   BBox: [234, 145, 456, 389] (x1, y1, x2, y2)
   Size: 222x244

2. Bus (92.1%)
   BBox: [678, 123, 987, 567] (x1, y1, x2, y2)
   Size: 309x444

3. Motorcycle (76.5%)
   BBox: [123, 456, 234, 678] (x1, y1, x2, y2)
   Size: 111x222
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💾 Saving output image...
  ✅ Saved detections to: test_car_detections.txt

✅ Done!
```

### Batch Processing

Process multiple images:

```bash
# Windows (PowerShell)
Get-ChildItem test_images\*.jpg | ForEach-Object {
    cargo run --release --bin test -- --image $_.FullName --conf 0.25
}


### Key Components

1. **Backbone (CSPDarknet53)**
   - Extract features at multiple scales
   - Cross Stage Partial connections

2. **Neck (FPN/PAN)**
   - Feature Pyramid Network (top-down)
   - Path Aggregation Network (bottom-up)

3. **Detection Head**
   - Regression: Bounding box coordinates (DFL)
   - Classification: Object class probabilities

4. **Loss Function**
   - Box Loss: IoU-based (CIoU/DIoU)
   - Class Loss: Binary Cross Entropy
   - DFL Loss: Distribution Focal Loss

---

## ⚙️ Configuration

### Training Config (`configs/train_config.yaml`)

```yaml
# Critical parameters
learning_rate: 0.001      # MOST IMPORTANT! Don't change unless you know what you're doing
epochs: 50                # More epochs = better (but watch for overfitting)
batch_size: 8             # Increase if you have more RAM

# Model
img_size: 640             # Input image size (640 is standard)
num_classes: 5            # Number of object classes
reg_max: 16               # DFL bins (default: 16)

# Loss weights (usually don't need to change)
box_loss_weight: 7.5
cls_loss_weight: 0.5
dfl_loss_weight: 1.5

# Early stopping
patience: 10              # Stop if no improvement for N epochs
min_delta: 0.001          # Minimum improvement to count
```

### Model Config (Auto-generated)

Located in checkpoint folder `runs/train/best/config.json`:

```json
{
  "model_type": "YOLOv8",
  "num_classes": 5,
  "reg_max": 16,
  "img_size": 640,
  "checkpoint_name": "best"
}
```

---

## 🐛 Troubleshooting

### Training Issues

#### 1. NaN Loss / Model Explosion

**Symptom:**
```
Epoch [1/10]
⚠️ NaN/Inf loss detected at batch 2
Train Loss: 0.0000, Val Loss: inf
```

**Solution:**
- ✅ **Check learning rate!** Should be `0.001`, NOT `1.5`
- Edit `configs/train_config.yaml`
- Delete `runs/train/` and restart training

#### 2. Val Loss Much Higher Than Train Loss

**Symptom:**
```
Train Loss: 0.62, Val Loss: 22.66
```

**Solution:**
- ✅ Use latest `trainer.rs` (validation should use same loss as training)
- ✅ Check if validation dataset has labels
- ✅ Lower confidence threshold during inference

#### 3. No Objects Detected

**Symptom:**
```
❌ No objects detected!
```

**Solutions:**
- ✅ Lower confidence: `--conf 0.1`
- ✅ Train longer (50+ epochs)
- ✅ Check if model trained properly (train loss < 1.0)
- ✅ Verify dataset labels are correct

#### 4. Out of Memory (OOM)

**Symptom:**
```
thread 'main' panicked at 'allocation failed'
```

**Solution:**
- ✅ Reduce batch size: `batch_size: 1` in config
- ✅ Reduce image size: `img_size: 416`
- ✅ Close other applications

### Compilation Issues

#### 1. Dependency Conflicts

```bash
# Clean and rebuild
cargo clean
cargo build --release
```

#### 2. Version Mismatch

```bash
# Update Rust
rustup update

# Update dependencies
cargo update
```

## 📚 Resources

### Official Documentation

- **Burn Framework:** [burn.dev](https://burn.dev/)
- **Rust Book:** [doc.rust-lang.org/book](https://doc.rust-lang.org/book/)
- **YOLOv8 Paper:** [arXiv:2305.09972](https://arxiv.org/abs/2305.09972)

### Tutorials & Guides

- **Rust ML Intro:** [www.arewelearningyet.com](https://www.arewelearningyet.com/)
- **YOLO Explained:** [pjreddie.com/darknet/yolo](https://pjreddie.com/darknet/yolo/)
- **Object Detection Guide:** [paperswithcode.com/task/object-detection](https://paperswithcode.com/task/object-detection)

### Datasets

- **Kaggle Dataset:** [Kaggle](https://www.kaggle.com/datasets/abdallahwagih/cars-detection)

### Community

- **Rust ML Community:** [discord.gg/rust-ml](https://discord.gg/rust-ml)
- **Burn Discord:** [discord.gg/uPEBbYYDB6](https://discord.gg/uPEBbYYDB6)
- **Stack Overflow:** Tag `rust` + `machine-learning`

### Blog Posts & Articles

- **Building ML in Rust:** [huggingface.co/blog/rust-ml](https://huggingface.co/blog/rust-ml)
- **Burn Deep Dive:** [burn.dev/blog](https://burn.dev/blog/)
- **YOLO Evolution:** [blog.roboflow.com/yolo-evolution](https://blog.roboflow.com/yolo-evolution/)

### YouTube Channels

- **Rust Programming:** [youtube.com/@NoBoilerplate](https://www.youtube.com/@NoBoilerplate)
- **Computer Vision:** [youtube.com/@FirstPrinciplesofComputerVision](https://www.youtube.com/@FirstPrinciplesofComputerVision)
- **YOLOv8 Tutorial:** [youtube.com/@ultralytics](https://www.youtube.com/@ultralytics)
