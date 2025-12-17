use burn::prelude::*;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, PaddingConfig2d};
use burn::tensor::activation;

#[derive(Module, Debug)]
pub struct Conv<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B>,
}

impl<B: Backend> Conv<B> {
    pub fn new(
        device: &B::Device,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
    ) -> Self {
        let padding = if kernel_size == 3 { 1 } else { 0 };

        Self {
            conv: Conv2dConfig::new([in_channels, out_channels], [kernel_size, kernel_size])
                .with_stride([stride, stride])
                .with_padding(PaddingConfig2d::Explicit(padding, padding))
                .with_bias(false)
                .init(device),
            bn: BatchNormConfig::new(out_channels).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        let x = self.bn.forward(x);
        activation::silu(x)
    }
}