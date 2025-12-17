use burn::prelude::*;

#[derive(Module, Debug, Clone)]
pub struct Upsample2d {
    scale_factor: usize,
}

impl Upsample2d {
    pub fn new(scale_factor: usize) -> Self {
        Self { scale_factor }
    }

    /// Nearest-neighbor upsample using reshape and repeat
    /// [B, C, H, W] -> [B, C, H*scale, W*scale]
    pub fn forward<B: Backend>(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, channels, height, width] = x.dims();
        
        // Method: Reshape + repeat (efficient nearest-neighbor)
        // [B, C, H, W] -> [B, C, H, 1, W, 1]
        let x = x.reshape([batch, channels, height, 1, width, 1]);
        
        // Repeat along dimensions 3 and 5
        // [B, C, H, 1, W, 1] -> [B, C, H, scale, W, scale]
        let x = x.repeat_dim(3, self.scale_factor);
        let x = x.repeat_dim(5, self.scale_factor);
        
        // Reshape back to [B, C, H*scale, W*scale]
        x.reshape([batch, channels, height * self.scale_factor, width * self.scale_factor])
    }
}