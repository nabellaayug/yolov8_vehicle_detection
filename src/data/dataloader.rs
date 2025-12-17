use burn::prelude::*;
use crate::data::dataset::{YoloDataset, BoundingBox};
use rand::seq::SliceRandom;

pub struct YoloDataLoader<B: Backend> {
    dataset: YoloDataset,
    batch_size: usize,
    shuffle: bool,
    device: B::Device,
    indices: Vec<usize>,
    current_idx: usize,
}

impl<B: Backend> YoloDataLoader<B> {
    pub fn new(
        dataset: YoloDataset,
        batch_size: usize,
        shuffle: bool,
        device: B::Device,
    ) -> Self {
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        
        if shuffle {
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
        }
        
        Self {
            dataset,
            batch_size,
            shuffle,
            device,
            indices,
            current_idx: 0,
        }
    }
    
    pub fn reset(&mut self) {
        self.current_idx = 0;
        if self.shuffle {
            let mut rng = rand::thread_rng();
            self.indices.shuffle(&mut rng);
        }
    }
    
    pub fn len(&self) -> usize {
        (self.dataset.len() + self.batch_size - 1) / self.batch_size
    }
    
    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }
}

pub struct YoloBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub boxes: Vec<Vec<BoundingBox>>,
    pub batch_size: usize,
}

impl<B: Backend> Iterator for YoloDataLoader<B> {
    type Item = YoloBatch<B>;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.dataset.len() {
            return None;
        }
        
        let end_idx = (self.current_idx + self.batch_size).min(self.dataset.len());
        let batch_indices = &self.indices[self.current_idx..end_idx];
        let actual_batch_size = batch_indices.len();
        
        let mut images_vec = Vec::new();
        let mut all_boxes = Vec::new();
        
        for &idx in batch_indices {
            if let Ok((img, boxes)) = self.dataset.get(idx) {
                let img = img.resize_exact(
                    self.dataset.img_size as u32,
                    self.dataset.img_size as u32,
                    image::imageops::FilterType::Lanczos3,
                );
                
                let rgb_img = img.to_rgb8();
                
                for c in 0..3 {
                    for y in 0..self.dataset.img_size {
                        for x in 0..self.dataset.img_size {
                            let pixel = rgb_img.get_pixel(x as u32, y as u32);
                            let val = pixel[c] as f32 / 255.0;
                            images_vec.push(val);
                        }
                    }
                }
                
                all_boxes.push(boxes);
            }
        }
        
        let images = Tensor::<B, 4>::from_floats(images_vec.as_slice(), &self.device)
            .reshape([actual_batch_size, 3, self.dataset.img_size, self.dataset.img_size]);
        
        self.current_idx = end_idx;
        
        Some(YoloBatch {
            images,
            boxes: all_boxes,
            batch_size: actual_batch_size,
        })
    }
}