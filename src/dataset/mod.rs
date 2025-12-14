pub mod preprocessing;

use burn::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

pub use preprocessing::{BBox, ImageData, ImagePreprocessor};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub data_path: String,
    pub image_size: usize,
    pub batch_size: usize,
    pub train_split: f32, // 0.7 = 70% train
    pub val_split: f32,   // 0.15 = 15% val
    pub test_split: f32,  // 0.15 = 15% test
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            data_path: "./data/processed".to_string(),
            image_size: 640,
            batch_size: 8,
            train_split: 0.7,
            val_split: 0.15,
            test_split: 0.15,
        }
    }
}

/// Road Vehicle Dataset dari Kaggle
/// Structure:
/// data/raw/
/// ├── images/
/// │   ├── car/
/// │   ├── bus/
/// │   ├── truck/
/// │   └── motorcycle/
#[derive(Debug)]
pub struct RoadVehicleDataset {
    config: DatasetConfig,
    train_images: Vec<(String, String)>, // (class_name, path)
    val_images: Vec<(String, String)>,
    test_images: Vec<(String, String)>,
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
            class_names: vec![
                "car".to_string(),
                "bus".to_string(),
                "truck".to_string(),
                "motorcycle".to_string(),
            ],
            class_to_id: std::collections::HashMap::new(),
        }
    }

    /// Load dataset dari folder structure dengan train/val/test split
    /// Expects: data/raw/images/{class_name}/{image.jpg}
    pub fn load_from_folder(&mut self, base_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut all_images: Vec<(String, String)> = Vec::new(); // (class, path)

        // Build class_to_id mapping
        for (id, class_name) in self.class_names.iter().enumerate() {
            self.class_to_id.insert(class_name.clone(), id);
        }

        // Scan folders untuk setiap class
        for class_name in &self.class_names.clone() {
            let class_path = format!("{}/images/{}", base_path, class_name);

            if !Path::new(&class_path).exists() {
                println!("⚠️  Class folder tidak ditemukan: {}", class_path);
                continue;
            }

            for entry in WalkDir::new(&class_path).into_iter().filter_map(|e| e.ok()) {
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if let Some(ext_str) = ext.to_str() {
                        if matches!(ext_str.to_lowercase().as_str(), "jpg" | "jpeg" | "png") {
                            all_images
                                .push((class_name.clone(), path.to_string_lossy().to_string()));
                        }
                    }
                }
            }
        }

        println!("✅ Found {} images total", all_images.len());

        // Shuffle images
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        all_images.shuffle(&mut rng);

        // Split train/val/test
        let total = all_images.len();
        let train_end = (total as f32 * self.config.train_split) as usize;
        let val_end = train_end + (total as f32 * self.config.val_split) as usize;

        self.train_images = all_images[..train_end].to_vec();
        self.val_images = all_images[train_end..val_end].to_vec();
        self.test_images = all_images[val_end..].to_vec();

        println!("📊 Dataset Split:");
        println!(
            "  Train: {} ({:.1}%)",
            self.train_images.len(),
            (self.train_images.len() as f32 / total as f32) * 100.0
        );
        println!(
            "  Val:   {} ({:.1}%)",
            self.val_images.len(),
            (self.val_images.len() as f32 / total as f32) * 100.0
        );
        println!(
            "  Test:  {} ({:.1}%)",
            self.test_images.len(),
            (self.test_images.len() as f32 / total as f32) * 100.0
        );

        Ok(())
    }

    pub fn get_train_images(&self) -> Vec<String> {
        self.train_images
            .iter()
            .map(|(_, path)| path.clone())
            .collect()
    }

    pub fn get_val_images(&self) -> Vec<String> {
        self.val_images
            .iter()
            .map(|(_, path)| path.clone())
            .collect()
    }

    pub fn get_test_images(&self) -> Vec<String> {
        self.test_images
            .iter()
            .map(|(_, path)| path.clone())
            .collect()
    }

    pub fn get_train_with_labels(&self) -> Vec<(String, usize)> {
        self.train_images
            .iter()
            .map(|(class, path)| (path.clone(), self.class_to_id[class]))
            .collect()
    }

    pub fn get_val_with_labels(&self) -> Vec<(String, usize)> {
        self.val_images
            .iter()
            .map(|(class, path)| (path.clone(), self.class_to_id[class]))
            .collect()
    }

    pub fn get_test_with_labels(&self) -> Vec<(String, usize)> {
        self.test_images
            .iter()
            .map(|(class, path)| (path.clone(), self.class_to_id[class]))
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

    /// Load batch dari image paths
    pub fn load_batch<B: Backend>(
        &self,
        image_paths: &[String],
        device: &B::Device,
    ) -> Result<Tensor<B, 4>, Box<dyn std::error::Error>> {
        let batch_size = self.config.batch_size;
        let mut batch_data = Vec::new();

        // Load images
        for path in image_paths.iter().take(batch_size) {
            match self.preprocessor.preprocess_image(Path::new(path)) {
                Ok(image_data) => {
                    batch_data.push(image_data.image_tensor);
                }
                Err(e) => {
                    eprintln!("⚠️  Error loading {}: {}", path, e);
                    // Create dummy tensor jika error
                    batch_data.push(vec![
                        0.0;
                        self.config.image_size * self.config.image_size * 3
                    ]);
                }
            }
        }

        // Pad batch jika kurang
        while batch_data.len() < batch_size {
            batch_data.push(vec![
                0.0;
                self.config.image_size * self.config.image_size * 3
            ]);
        }

        // Convert ke tensor
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

    /// Load batch dengan labels
    pub fn load_batch_with_labels<B: Backend>(
        &self,
        image_paths_with_labels: &[(String, usize)],
        device: &B::Device,
    ) -> Result<(Tensor<B, 4>, Vec<usize>), Box<dyn std::error::Error>> {
        let batch_size = self.config.batch_size;
        let mut batch_data = Vec::new();
        let mut labels = Vec::new();

        // Load images dengan labels
        for (path, label) in image_paths_with_labels.iter().take(batch_size) {
            match self.preprocessor.preprocess_image(Path::new(path)) {
                Ok(image_data) => {
                    batch_data.push(image_data.image_tensor);
                    labels.push(*label);
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

        // Pad batch
        while batch_data.len() < batch_size {
            batch_data.push(vec![
                0.0;
                self.config.image_size * self.config.image_size * 3
            ]);
            labels.push(0);
        }

        // Convert ke tensor
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
        assert_eq!(dataset.num_classes(), 4);
    }

    #[test]
    fn test_class_mapping() {
        let dataset = RoadVehicleDataset::new(DatasetConfig::default());
        assert_eq!(dataset.class_to_id("car"), Some(0));
        assert_eq!(dataset.class_to_id("bus"), Some(1));
        assert_eq!(dataset.class_to_id("truck"), Some(2));
        assert_eq!(dataset.class_to_id("motorcycle"), Some(3));
    }

    #[test]
    fn test_split_ratios() {
        let config = DatasetConfig {
            data_path: "./data".to_string(),
            image_size: 640,
            batch_size: 8,
            train_split: 0.7,
            val_split: 0.15,
            test_split: 0.15,
        };

        assert_eq!(
            config.train_split + config.val_split + config.test_split,
            1.0
        );
    }
}
