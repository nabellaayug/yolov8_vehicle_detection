use anyhow::Result;
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    pub path: String,
    pub train: String,
    pub test: String,
    pub val: String,
    pub nc: usize,
    pub names: Vec<String>,
}

impl DataConfig {
    pub fn from_yaml(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: DataConfig = serde_yaml::from_str(&content)?;
        Ok(config)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub class_id: usize,
    pub x_center: f32,
    pub y_center: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Clone)]
pub struct YoloDataset {
    samples: Vec<(PathBuf, PathBuf)>,
    pub num_classes: usize,
    pub class_names: Vec<String>,
    pub img_size: usize,
}

impl YoloDataset {
    /// Load dataset from data.yaml
    /// Expected structure:
    /// Cars Detection/
    /// ├── images/
    /// │   ├── train/
    /// │   ├── test/
    /// │   └── val/
    /// ├── labels/
    /// │   ├── train/
    /// │   ├── test/
    /// │   └── val/
    /// └── data.yaml
    pub fn new(yaml_path: &str, split: &str, img_size: usize) -> Result<Self> {
        let config = DataConfig::from_yaml(yaml_path)?;

        let base_path = PathBuf::from(yaml_path).parent().unwrap().to_path_buf();

        // Validate split name
        let split_name = match split.to_lowercase().as_str() {
            "train" => "train",
            "test" => "test",
            "val" => "val",
            _ => {
                return Err(anyhow::anyhow!(
                    "❌ Unknown split: {}. Use 'train', 'test', or 'val'",
                    split
                ))
            }
        };

        // New structure: images/{split}/ and labels/{split}/
        let img_dir = base_path.join("images").join(split_name);
        let label_dir = base_path.join("labels").join(split_name);

        if !img_dir.exists() {
            return Err(anyhow::anyhow!(
                "❌ Image directory not found: {}",
                img_dir.display()
            ));
        }

        if !label_dir.exists() {
            return Err(anyhow::anyhow!(
                "❌ Label directory not found: {}",
                label_dir.display()
            ));
        }

        let mut samples = Vec::new();

        if let Ok(entries) = std::fs::read_dir(&img_dir) {
            for entry in entries.flatten() {
                let img_path = entry.path();

                if img_path.is_file() {
                    if let Some(ext) = img_path.extension() {
                        if matches!(ext.to_str(), Some("jpg") | Some("jpeg") | Some("png")) {
                            // Find corresponding label file in labels/{split}/
                            let label_path = label_dir
                                .join(img_path.file_stem().unwrap())
                                .with_extension("txt");

                            if label_path.exists() {
                                samples.push((img_path, label_path));
                            } else {
                                eprintln!(
                                    "⚠️  Warning: Label not found for {}",
                                    img_path.display()
                                );
                            }
                        }
                    }
                }
            }
        }

        println!("✅ Loaded {} {} samples", samples.len(), split_name);

        if samples.is_empty() {
            return Err(anyhow::anyhow!(
                "❌ No samples found in {}. Check if images and labels exist.",
                img_dir.display()
            ));
        }

        Ok(Self {
            samples,
            num_classes: config.nc,
            class_names: config.names,
            img_size,
        })
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn get(&self, idx: usize) -> Result<(DynamicImage, Vec<BoundingBox>)> {
        if idx >= self.samples.len() {
            return Err(anyhow::anyhow!(
                "❌ Index {} out of bounds. Dataset has {} samples",
                idx,
                self.samples.len()
            ));
        }

        let (img_path, label_path) = &self.samples[idx];

        let img = image::open(img_path)?;
        let label_content = std::fs::read_to_string(label_path)?;

        let mut boxes = Vec::new();
        for line in label_content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 5 {
                match (
                    parts[0].parse::<usize>(),
                    parts[1].parse::<f32>(),
                    parts[2].parse::<f32>(),
                    parts[3].parse::<f32>(),
                    parts[4].parse::<f32>(),
                ) {
                    (Ok(class_id), Ok(x_center), Ok(y_center), Ok(width), Ok(height)) => {
                        boxes.push(BoundingBox {
                            class_id,
                            x_center,
                            y_center,
                            width,
                            height,
                        });
                    }
                    _ => {
                        eprintln!(
                            "⚠️  Warning: Invalid label format in {}: {}",
                            label_path.display(),
                            line
                        );
                    }
                }
            }
        }

        Ok((img, boxes))
    }
}

/// Reorganize dataset from old structure to new structure
/// From: train/images, train/labels, test/images, val/images
/// To:   images/train, labels/train, images/test, labels/test, images/val, labels/val
pub fn reorganize_dataset(dataset_dir: &str) -> Result<()> {
    let base_path = std::path::Path::new(dataset_dir);
    if !base_path.exists() {
        return Err(anyhow::anyhow!("❌ Path {} tidak ditemukan!", dataset_dir));
    }

    println!("🔄 Reorganizing dataset structure...");
    println!("From: train/images, train/labels, test/images, val/images");
    println!(
        "To:   images/train, labels/train, images/test, labels/test, images/val, labels/val\n"
    );

    println!("📁 Creating new directory structure...");

    let splits = vec!["train", "test", "val"];
    for split in &splits {
        fs::create_dir_all(base_path.join("images").join(split))?;
        fs::create_dir_all(base_path.join("labels").join(split))?;
    }
    println!("  ✅ Created images/ and labels/ directories\n");

    // Move files for each split
    for split in &splits {
        reorganize_split(base_path, split)?;
    }

    // Create data.yaml
    println!("📝 Creating data.yaml...");
    let yaml_content = r#"path: ./data/Cars Detection
train: images/train
test: images/test
val: images/val

nc: 5
names: ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']
"#;

    let yaml_path = base_path.join("data.yaml");
    fs::write(&yaml_path, yaml_content)?;
    println!("  ✅ Saved to {}\n", yaml_path.display());

    // Clean up old directories
    println!("🧹 Cleaning up old directory structure...");
    for split in &splits {
        let old_split_dir = base_path.join(split);
        if old_split_dir.exists() {
            fs::remove_dir_all(&old_split_dir)?;
            println!("  ✅ Removed old {}/", split);
        }
    }

    println!("\n✅ Dataset reorganization completed!");
    println!("\nNew structure:");
    println!("  Cars Detection/");
    println!("  ├── images/");
    println!("  │   ├── train/");
    println!("  │   ├── test/");
    println!("  │   └── val/");
    println!("  ├── labels/");
    println!("  │   ├── train/");
    println!("  │   ├── test/");
    println!("  │   └── val/");
    println!("  └── data.yaml");

    Ok(())
}

fn reorganize_split(base_path: &std::path::Path, split: &str) -> Result<()> {
    println!("📋 Organizing {} split...", split);

    let old_split_dir = base_path.join(split);
    let new_img_dir = base_path.join("images").join(split);
    let new_label_dir = base_path.join("labels").join(split);

    if !old_split_dir.exists() {
        println!("  ⚠️  {} folder tidak ditemukan, skip", split);
        return Ok(());
    }

    let mut img_count = 0;
    let mut label_count = 0;

    // Move images
    let old_img_dir = old_split_dir.join("images");
    if old_img_dir.exists() {
        if let Ok(entries) = fs::read_dir(&old_img_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    if let Some(file_name) = path.file_name() {
                        let dest = new_img_dir.join(file_name);
                        fs::copy(&path, &dest)?;
                        img_count += 1;
                    }
                }
            }
        }
    }

    // Move labels
    let old_label_dir = old_split_dir.join("labels");
    if old_label_dir.exists() {
        if let Ok(entries) = fs::read_dir(&old_label_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    if let Some(file_name) = path.file_name() {
                        let dest = new_label_dir.join(file_name);
                        fs::copy(&path, &dest)?;
                        label_count += 1;
                    }
                }
            }
        }
    }

    println!("  ✅ {} images, {} labels moved", img_count, label_count);

    Ok(())
}
