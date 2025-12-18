use super::Conv;
use burn::{
    nn::pool::{MaxPool2d, MaxPool2dConfig},
    prelude::*,
};

#[derive(Module, Debug)]
pub struct SPPF<B: Backend> {
    conv1: Conv<B>,
    conv2: Conv<B>,
    pool: MaxPool2d,
}

impl<B: Backend> SPPF<B> {
    pub fn new(device: &B::Device, in_channels: usize, out_channels: usize) -> Self {
        let hidden = in_channels / 2;
        
        // First conv: in_channels -> hidden (1x1 kernel) - channel reduction
        let conv1 = Conv::new(device, in_channels, hidden, 1, 1);

        // Second conv: hidden * 4 -> out_channels (1x1 kernel)
        // After concatenating 4 tensors of size [B, hidden, H, W], we get [B, hidden*4, H, W]
        let conv2 = Conv::new(device, hidden * 4, out_channels, 1, 1);

        let pool = MaxPool2dConfig::new([5, 5])
            .with_strides([1, 1])
            .with_padding(nn::PaddingConfig2d::Explicit(2, 2))
            .init();

        Self { conv1, conv2, pool }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // Initial processing: reduce channels
        let x = self.conv1.forward(x);  // [B, hidden, H, W]

        // Apply max pooling 3 times to create multi-scale features
        let p1 = self.pool.forward(x.clone());  // [B, hidden, H, W]
        let p2 = self.pool.forward(p1.clone()); // [B, hidden, H, W]
        let p3 = self.pool.forward(p2.clone()); // [B, hidden, H, W]

        // Concatenate: [x, p1, p2, p3] -> [B, hidden*4, H, W]
        let cat = Tensor::cat(vec![x, p1, p2, p3], 1);
        
        // Final conv: hidden*4 -> out_channels
        self.conv2.forward(cat)
    }
}