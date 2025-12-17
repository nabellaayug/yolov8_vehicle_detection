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
        // First conv: in_channels -> out_channels (1x1 kernel)
        let conv1 = Conv::new(device, in_channels, out_channels, 1, 1);

        // Second conv: out_channels -> out_channels (1x1 kernel)
        let conv2 = Conv::new(device, hidden * 4, out_channels, 1, 1);

        let pool = MaxPool2dConfig::new([5, 5])
            .with_strides([1, 1])
            .with_padding(nn::PaddingConfig2d::Explicit(2, 2))
            .init();

        Self { conv1, conv2, pool }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // Initial processing
        let x = self.conv1.forward(x);

        let p1 = self.pool.forward(x.clone());
        let p2 = self.pool.forward(p1.clone());
        let p3 = self.pool.forward(p2.clone());

        let cat = Tensor::cat(vec![x, p1, p2, p3], 1);
        self.conv2.forward(cat)
    }
}
