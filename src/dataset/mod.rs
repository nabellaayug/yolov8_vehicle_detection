pub mod preprocessing;

use burn::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

pub use preprocessing::{BBox, ImageData, ImagePreprocessor};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub data_path: String,
    pub image_size: usize,
    pub batch_size: usize,
    pub train_split: f32,
    pub val_split: f32,
    pub test_split: f32,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            data_path: "./data/raw".to_string(),
            image_size: 640,
            batch_size: 8,
            train_split: 0.7,
            val_split: 0.2,
            test_split: 0.1,
        }
    }
}

/// Cars Detection Dataset dari Kaggle (sudah ter-split)
/// Structure:
/// data/raw/
/// ├── images/
/// │   ├── train/
/// │   ├── val/
/// │   └── test/
/// └── labels/
///     ├── train/
///     ├── val/
///     └── test/
#[derive(Debug)]
pub struct RoadVehicleDataset {
    config: DatasetConfig,
    train_images: Vec<(PathBuf, Option<PathBuf>)>,
    val_images: Vec<(PathBuf, Option<PathBuf>)>,
    test_images: Vec<(PathBuf, Option<PathBuf>)>,
    class_names: Vec<String>,
    class_to_id: std::collections::HashMap<String, usize>,
}

impl RoadVehicleDataset {
    pub fn new(config: DatasetConfig) -> Self {
        Self {
            config,
            train_images: Vec::new(),
            val_images: Vec::new(),
            test_images: Vec::new(),
            class_names: vec!["car".to_string()],
            class_to_id: std::collections::HashMap::new(),
        }
    }

    /// Organize dataset dari Kaggle structure, kemudian load
    pub fn organize_and_load(
        &mut self,
        kaggle_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n🚀 Organizing Cars Detection Dataset");
        println!("====================================\n");

        // Create output directory structure
        println!("📁 Creating output directory structure...");
        Self::create_directories(&self.config.data_path)?;
        println!("✅ Directories created\n");

        // Copy dataset splits
        println!("📋 Copying dataset splits...\n");

        let train_img = Self::copy_split(kaggle_path, &self.config.data_path, "train")?;
        println!("  ✅ Train: {} images\n", train_img);

        let val_img = Self::copy_split(kaggle_path, &self.config.data_path, "val")?;
        println!("  ✅ Val: {} images\n", val_img);

        let test_img = Self::copy_split(kaggle_path, &self.config.data_path, "test")?;
        println!("  ✅ Test: {} images\n", test_img);

        let total = train_img + val_img + test_img;
        if total == 0 {
            return Err("No images found in dataset!".into());
        }

        println!("{}", "=".repeat(70));
        println!("📊 DATASET ORGANIZATION SUMMARY");
        println!("{}", "=".repeat(70));
        println!("Total images: {}", total);
        println!();
        println!(
            "  Train: {} ({:.1}%)",
            train_img,
            (train_img as f32 / total as f32) * 100.0
        );
        println!(
            "  Val:   {} ({:.1}%)",
            val_img,
            (val_img as f32 / total as f32) * 100.0
        );
        println!(
            "  Test:  {} ({:.1}%)",
            test_img,
            (test_img as f32 / total as f32) * 100.0
        );
        println!("{}", "=".repeat(70));
        println!();

        // Now load the organized dataset
        let base_path = self.config.data_path.clone();
        self.load_from_folder(&base_path)?;

        Ok(())
    }

    /// Load dataset dari folder structure train/val/test
    pub fn load_from_folder(&mut self, base_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("📂 Loading organized dataset...");

    // Build class_to_id mapping
    self.class_to_id.clear();
    for (id, class_name) in self.class_names.iter().enumerate() {
        self.class_to_id.insert(class_name.clone(), id);
    }

    let splits = ["train", "val", "test"];

    for split in splits.iter() {
        let images_dir = format!("{}/images/{}", base_path, split);
        let labels_dir = format!("{}/labels/{}", base_path, split);

        println!("   📂 Scanning {}: {}", split, images_dir);

        if !Path::new(&images_dir).exists() {
            println!("   ❌ Images directory not found: {}", images_dir);
            continue;
        }

        let mut split_images: Vec<(PathBuf, Option<PathBuf>)> = Vec::new();
        let mut found_count = 0;

        for entry in WalkDir::new(&images_dir).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }

            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if !matches!(ext.to_lowercase().as_str(), "jpg" | "jpeg" | "png") {
                continue;
            }

            found_count += 1;
            let img_path = path.to_path_buf();

            let label_path = {
                let label_file = Path::new(&labels_dir)
                    .join(format!("{}.txt", path.file_stem().unwrap().to_string_lossy()));
                if label_file.exists() {
                    Some(label_file)
                } else {
                    None
                }
            };

            split_images.push((img_path, label_path));
        }

        println!("   ✓ Found {} images", found_count);

        match *split {
            "train" => self.train_images = split_images,
            "val" => self.val_images = split_images,
            "test" => self.test_images = split_images,
            _ => {}
        }
    }

    let total = self.train_images.len() + self.val_images.len() + self.test_images.len();
    println!("✅ Loaded {} total images\n", total);

    println!("📊 Dataset Split:");
    println!(
        "  Train: {} ({:.1}%)",
        self.train_images.len(),
        Self::percent(self.train_images.len(), total)
    );
    println!(
        "  Val:   {} ({:.1}%)",
        self.val_images.len(),
        Self::percent(self.val_images.len(), total)
    );
    println!(
        "  Test:  {} ({:.1}%)",
        self.test_images.len(),
        Self::percent(self.test_images.len(), total)
    );

    Ok(())
}

fn percent(part: usize, total: usize) -> f32 {
    if total == 0 {
        0.0
    } else {
        (part as f32 / total as f32) * 100.0
    }
}


    // =====================================================
    // Private helper functions untuk organize_and_load
    // =====================================================

    fn create_directories(base_path: &str) -> io::Result<()> {
        let dirs = vec![
            format!("{}/images/train", base_path),
            format!("{}/images/val", base_path),
            format!("{}/images/test", base_path),
            format!("{}/labels/train", base_path),
            format!("{}/labels/val", base_path),
            format!("{}/labels/test", base_path),
        ];

        for dir in dirs {
            fs::create_dir_all(&dir)?;
        }

        Ok(())
    }

    fn copy_split(
        kaggle_path: &str,
        output_base: &str,
        split: &str,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let kaggle_images_dir = format!("{}/{}/images", kaggle_path, split);
        let kaggle_labels_dir = format!("{}/{}/labels", kaggle_path, split);

        let output_images_dir = format!("{}/images/{}", output_base, split);
        let output_labels_dir = format!("{}/labels/{}", output_base, split);

        let mut images_count = 0;

        println!("   📂 Processing {}", split);
        println!("      From: {}", kaggle_images_dir);
        println!("      To:   {}", output_images_dir);

        // Check if source directory exists
        if !Path::new(&kaggle_images_dir).exists() {
            eprintln!("   ❌ Source directory not found: {}", kaggle_images_dir);
            return Ok(0);
        }

        println!("   ✓ Source directory found");
        println!("   📂 Source path: {}", kaggle_images_dir);
        println!("   📂 Output path: {}", output_images_dir);

        // List what's in source directory first
        match std::fs::read_dir(&kaggle_images_dir) {
            Ok(entries) => {
                let file_list: Vec<_> = entries
                    .filter_map(|e| e.ok())
                    .map(|e| e.file_name())
                    .collect();
                println!("   ℹ️  Files in source: {:?}", file_list.len());
                if file_list.is_empty() {
                    println!("   ⚠️  Source directory is EMPTY!");
                }
            }
            Err(e) => {
                println!("   ❌ Cannot read source directory: {}", e);
            }
        }

        // Copy images
        let mut file_count = 0;
        for entry in WalkDir::new(&kaggle_images_dir)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if let Some(ext_str) = ext.to_str() {
                        if matches!(ext_str.to_lowercase().as_str(), "jpg" | "jpeg" | "png") {
                            file_count += 1;
                            let file_name = path.file_name().unwrap();
                            let dest_path =
                                format!("{}/{}", output_images_dir, file_name.to_string_lossy());

                            match fs::copy(path, &dest_path) {
                                Ok(_) => {
                                    images_count += 1;
                                    // Also copy corresponding label if exists
                                    if Path::new(&kaggle_labels_dir).exists() {
                                        let label_name = format!(
                                            "{}.txt",
                                            path.file_stem().unwrap().to_string_lossy()
                                        );
                                        let label_src =
                                            format!("{}/{}", kaggle_labels_dir, label_name);
                                        let label_dest =
                                            format!("{}/{}", output_labels_dir, label_name);

                                        if Path::new(&label_src).exists() {
                                            let _ = fs::copy(&label_src, &label_dest);
                                        }
                                    }
                                }
                                Err(e) => {
                                    eprintln!(
                                        "   ❌ Error copying {}: {}",
                                        file_name.to_string_lossy(),
                                        e
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        if file_count == 0 {
            println!("   ⚠️  No image files found in {}", kaggle_images_dir);
        } else {
            println!("   ✓ Copied {} / {} images", images_count, file_count);
        }

        Ok(images_count)
    }

    pub fn get_train_images(&self) -> Vec<String> {
        self.train_images
            .iter()
            .map(|(path, _)| path.to_string_lossy().to_string())
            .collect()
    }

    pub fn get_val_images(&self) -> Vec<String> {
        self.val_images
            .iter()
            .map(|(path, _)| path.to_string_lossy().to_string())
            .collect()
    }

    pub fn get_test_images(&self) -> Vec<String> {
        self.test_images
            .iter()
            .map(|(path, _)| path.to_string_lossy().to_string())
            .collect()
    }

    pub fn get_train_with_labels(&self) -> Vec<(String, Option<String>)> {
        self.train_images
            .iter()
            .map(|(img_path, label_path)| {
                (
                    img_path.to_string_lossy().to_string(),
                    label_path.as_ref().map(|p| p.to_string_lossy().to_string()),
                )
            })
            .collect()
    }

    pub fn get_val_with_labels(&self) -> Vec<(String, Option<String>)> {
        self.val_images
            .iter()
            .map(|(img_path, label_path)| {
                (
                    img_path.to_string_lossy().to_string(),
                    label_path.as_ref().map(|p| p.to_string_lossy().to_string()),
                )
            })
            .collect()
    }

    pub fn get_test_with_labels(&self) -> Vec<(String, Option<String>)> {
        self.test_images
            .iter()
            .map(|(img_path, label_path)| {
                (
                    img_path.to_string_lossy().to_string(),
                    label_path.as_ref().map(|p| p.to_string_lossy().to_string()),
                )
            })
            .collect()
    }

    pub fn num_classes(&self) -> usize {
        self.class_names.len()
    }

    pub fn class_names(&self) -> &[String] {
        &self.class_names
    }

    pub fn class_to_id(&self, class: &str) -> Option<usize> {
        self.class_to_id.get(class).copied()
    }

    pub fn id_to_class(&self, id: usize) -> Option<&str> {
        if id < self.class_names.len() {
            Some(&self.class_names[id])
        } else {
            None
        }
    }
}

/// Batch loader untuk efficient data loading
#[derive(Debug)]
pub struct BatchLoader {
    config: DatasetConfig,
    preprocessor: ImagePreprocessor,
}

impl BatchLoader {
    pub fn new(config: DatasetConfig) -> Self {
        let preprocessor = ImagePreprocessor::new(
            std::path::PathBuf::from(&config.data_path),
            config.image_size as u32,
        );

        Self {
            config,
            preprocessor,
        }
    }

    pub fn load_batch<B: Backend>(
        &self,
        image_paths: &[String],
        device: &B::Device,
    ) -> Result<Tensor<B, 4>, Box<dyn std::error::Error>> {
        let batch_size = self.config.batch_size;
        let mut batch_data = Vec::new();

        for path in image_paths.iter().take(batch_size) {
            match self.preprocessor.preprocess_image(Path::new(path)) {
                Ok(image_data) => {
                    batch_data.push(image_data.image_tensor);
                }
                Err(e) => {
                    eprintln!("⚠️  Error loading {}: {}", path, e);
                    batch_data.push(vec![
                        0.0;
                        self.config.image_size * self.config.image_size * 3
                    ]);
                }
            }
        }

        while batch_data.len() < batch_size {
            batch_data.push(vec![
                0.0;
                self.config.image_size * self.config.image_size * 3
            ]);
        }

        let h = self.config.image_size;
        let w = self.config.image_size;
        let mut tensor_data: Vec<f32> = Vec::new();

        for img_tensor in &batch_data[..batch_size] {
            tensor_data.extend(img_tensor);
        }

        let tensor = Tensor::<B, 1>::from_floats(tensor_data.as_slice(), device)
            .reshape([batch_size, 3, h, w]);

        Ok(tensor)
    }

    pub fn load_batch_with_labels<B: Backend>(
        &self,
        image_paths_with_labels: &[(String, Option<String>)],
        device: &B::Device,
    ) -> Result<(Tensor<B, 4>, Vec<usize>), Box<dyn std::error::Error>> {
        let batch_size = self.config.batch_size;
        let mut batch_data = Vec::new();
        let mut labels = Vec::new();

        for (path, _label_path) in image_paths_with_labels.iter().take(batch_size) {
            match self.preprocessor.preprocess_image(Path::new(path)) {
                Ok(image_data) => {
                    batch_data.push(image_data.image_tensor);
                    labels.push(0); // Car class
                }
                Err(e) => {
                    eprintln!("⚠️  Error loading {}: {}", path, e);
                    batch_data.push(vec![
                        0.0;
                        self.config.image_size * self.config.image_size * 3
                    ]);
                    labels.push(0);
                }
            }
        }

        while batch_data.len() < batch_size {
            batch_data.push(vec![
                0.0;
                self.config.image_size * self.config.image_size * 3
            ]);
            labels.push(0);
        }

        let h = self.config.image_size;
        let w = self.config.image_size;
        let mut tensor_data: Vec<f32> = Vec::new();

        for img_tensor in &batch_data[..batch_size] {
            tensor_data.extend(img_tensor);
        }

        let tensor = Tensor::<B, 1>::from_floats(tensor_data.as_slice(), device)
            .reshape([batch_size, 3, h, w]);

        Ok((tensor, labels))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_creation() {
        let config = DatasetConfig::default();
        let dataset = RoadVehicleDataset::new(config);
        assert_eq!(dataset.num_classes(), 1);
    }

    #[test]
    fn test_split_ratios() {
        let config = DatasetConfig {
            data_path: "./data".to_string(),
            image_size: 640,
            batch_size: 8,
            train_split: 0.7,
            val_split: 0.2,
            test_split: 0.1,
        };

        assert_eq!(
            config.train_split + config.val_split + config.test_split,
            1.0
        );
    }
}
