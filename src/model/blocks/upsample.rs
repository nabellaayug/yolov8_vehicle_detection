use burn::prelude::*;
use burn::tensor::ops::InterpolateMode;

#[derive(Module, Debug, Clone)]
pub struct Upsample2d {
    scale_factor: usize,
}

impl Upsample2d {
    pub fn new(scale_factor: usize) -> Self {
        Self { scale_factor }
    }

    pub fn forward<B: Backend>(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, channels, height, width] = x.dims();
        
        let new_height = (height as f32 * self.scale_factor as f32) as usize;
        let new_width = (width as f32 * self.scale_factor as f32) as usize;
        // Simple nearest neighbor upsampling
        // Reshape: [B, C, H, W] -> [B, C, H, 1, W, 1]
        let x = x.reshape([batch, channels, height, 1, width, 1]);
        
        // Repeat dengan array: [1, 1, 1, scale, 1, scale]
        let repeat_dims = [1, 1, 1, self.scale_factor, 1, self.scale_factor];
        let x = x.repeat(&repeat_dims);
        
        // Reshape back: [B, C, H*scale, W*scale]
        x.reshape([batch, channels, new_height, new_width])
    }
}