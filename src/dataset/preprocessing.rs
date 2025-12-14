use image::{io::Reader as ImageReader, RgbImage};
use std::path::{Path, PathBuf};
use std::fs;
use walkdir::WalkDir;
use serde::{Serialize, Deserialize};
use std::error::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BBox {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
    pub class: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    pub image_path: PathBuf,
    pub image_tensor: Vec<f32>,
    pub bboxes: Vec<BBox>,
    pub width: u32,
    pub height: u32,
}
#[derive(Debug)]
pub struct ImagePreprocessor {
    target_size: u32,
    dataset_path: PathBuf,
}

impl ImagePreprocessor {
    pub fn new(dataset_path: PathBuf, target_size: u32) -> Self {
        Self {
            target_size,
            dataset_path,
        }
    }

    /// Scan dataset dan load semua gambar
    pub fn load_dataset(&self) -> Result<Vec<PathBuf>, Box<dyn Error>> {
        let mut images = Vec::new();
        
        for entry in WalkDir::new(&self.dataset_path)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if let Some(ext_str) = ext.to_str() {
                    if matches!(ext_str.to_lowercase().as_str(), "jpg" | "jpeg" | "png") {
                        images.push(path.to_path_buf());
                    }
                }
            }
        }
        
        println!("✅ Found {} images", images.len());
        Ok(images)
    }

    /// Preprocess satu gambar
    pub fn preprocess_image(&self, image_path: &Path) -> Result<ImageData, Box<dyn Error>> {
        // Load image
        let img_reader = ImageReader::open(image_path)?;
        let img = img_reader.decode()?;
        let rgb_img = img.to_rgb8();
        
        let (orig_width, orig_height) = (rgb_img.width(), rgb_img.height());
        
        // Resize ke target size (dengan padding untuk maintain aspect ratio)
        let resized = self.resize_with_padding(&rgb_img, self.target_size);
        
        // Normalize ke [0, 1]
        let tensor = self.image_to_tensor(&resized);
        
        Ok(ImageData {
            image_path: image_path.to_path_buf(),
            image_tensor: tensor,
            bboxes: Vec::new(), // TODO: Load dari annotation
            width: orig_width,
            height: orig_height,
        })
    }

    /// Resize image dengan padding (maintain aspect ratio)
    fn resize_with_padding(&self, img: &RgbImage, target_size: u32) -> RgbImage {
        let (w, h) = img.dimensions();
        
        // Calculate scale untuk fit ke target size
        let scale = (target_size as f32 / w.max(h) as f32).min(1.0);
        
        let new_w = (w as f32 * scale) as u32;
        let new_h = (h as f32 * scale) as u32;
        
        // Resize
        let resized = image::imageops::resize(
            img,
            new_w,
            new_h,
            image::imageops::FilterType::Lanczos3,
        );
        
        // Create canvas dengan padding
        let mut canvas = RgbImage::from_pixel(
            target_size,
            target_size,
            image::Rgb([128, 128, 128]), // Gray padding
        );
        
        // Paste resized image di center
        let offset_x = (target_size - new_w) / 2;
        let offset_y = (target_size - new_h) / 2;
        
        image::imageops::overlay(&mut canvas, &resized, offset_x.into(), offset_y.into());
        
        canvas
    }

    /// Convert image ke tensor format (CHW normalized)
    fn image_to_tensor(&self, img: &RgbImage) -> Vec<f32> {
        let (w, h) = img.dimensions();
        let total_pixels = (w * h) as usize;
        let mut tensor = vec![0.0f32; total_pixels * 3];
        
        // Convert RGB ke CHW format dan normalize
        // Channel-first format: [R, G, B]
        for (idx, pixel) in img.pixels().enumerate() {
            tensor[0 * total_pixels + idx] = pixel[0] as f32 / 255.0; // R channel
            tensor[1 * total_pixels + idx] = pixel[1] as f32 / 255.0; // G channel
            tensor[2 * total_pixels + idx] = pixel[2] as f32 / 255.0; // B channel
        }
        
        tensor
    }

    /// Save preprocessed data ke file
    pub fn save_preprocessed(
        &self,
        data: &ImageData,
        output_dir: &Path,
    ) -> Result<(), Box<dyn Error>> {
        fs::create_dir_all(output_dir)?;
        
        // Save tensor as binary
        let filename = data.image_path
            .file_stem()
            .unwrap()
            .to_string_lossy();
        
        let tensor_path = output_dir.join(format!("{}.bin", filename));
        let binary_data: Vec<u8> = data.image_tensor
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        
        fs::write(tensor_path, binary_data)?;
        
        // Save metadata as JSON
        let meta_path = output_dir.join(format!("{}.json", filename));
        let meta = serde_json::json!({
            "width": data.width,
            "height": data.height,
            "tensor_size": data.image_tensor.len(),
            "bboxes": data.bboxes,
        });
        
        fs::write(meta_path, serde_json::to_string_pretty(&meta)?)?;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocessor_creation() {
        let preprocessor = ImagePreprocessor::new(
            PathBuf::from("./data"),
            640,
        );
        assert_eq!(preprocessor.target_size, 640);
    }
}