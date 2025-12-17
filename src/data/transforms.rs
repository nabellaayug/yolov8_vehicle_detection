use image::DynamicImage;
use rand::Rng;

pub struct DataAugmentation {
    pub enable: bool,
}

impl DataAugmentation {
    pub fn new(enable: bool) -> Self {
        Self { enable }
    }
    
    pub fn apply(&self, img: DynamicImage) -> DynamicImage {
        if !self.enable {
            return img;
        }
        
        let mut rng = rand::thread_rng();
        let mut img = img;
        
        // Random horizontal flip (50% chance)
        if rng.gen_bool(0.5) {
            img = img.fliph();
        }
        
        // Random brightness adjustment
        if rng.gen_bool(0.3) {
            let factor = rng.gen_range(0.8..1.2);
            img = img.brighten((factor * 20.0) as i32);
        }
        
        img
    }
}

// TODO: Implement more augmentations
// - Mosaic
// - MixUp
// - Random crop
// - HSV jitter